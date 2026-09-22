"""并行调度。

关于「用多进程还是多线程」——这是本模块最关键的取舍，结论是**以线程池为主**：

    直觉上 CPU 密集任务应该用多进程，但这里不成立，原因有三：
      1. 本项目的重计算全部落在会释放 GIL 的库里：torch 的卷积与矩阵运算、
         onnxruntime 的推理、numpy/scipy 的 FFT 都在 C 层执行并释放 GIL。
         线程能拿到真正的并行度。
      2. Windows 的进程创建是 spawn 语义，每个子进程都要重新 import torch
         并重建 ONNX 会话。实测启动开销在秒级，内存各占数百 MB，
         开 4 个 worker 的代价远超收益。
      3. 分离与转录要共享已加载的模型对象，进程间无法直接共享，
         只能各自重复加载。

因此采用「线程池 + 分级信号量」的结构：
    分离阶段用 sep_sem 限流（受显存与内存约束，通常只能 1~2 并发）；
    转录阶段用 tr_sem 限流（受 CPU 核心数约束，可开到 4~6 并发）。
两级分开限流的意义在于：分离与转录可以重叠进行 —— 文件 A 在转录时，
文件 B 可以同时在分离，从而把整机资源吃满，而不是串行等每一步。
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from .config import Config

__all__ = ["JobResult", "StageLimiter", "ParallelRunner"]


@dataclass
class JobResult:
    """单个任务的执行结果。"""

    job_id: str
    source: str
    ok: bool = False
    duration_sec: float = 0.0
    elapsed_sec: float = 0.0
    error: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    stage_timings: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "source": self.source,
            "ok": self.ok,
            "duration_sec": round(self.duration_sec, 2),
            "elapsed_sec": round(self.elapsed_sec, 2),
            "speedup_vs_realtime": round(self.duration_sec / self.elapsed_sec, 2)
            if self.elapsed_sec > 0 else None,
            "error": self.error,
            "stage_timings": {k: round(v, 2) for k, v in self.stage_timings.items()},
            "payload": self.payload,
        }


class StageLimiter:
    """带统计的分级并发限制器。"""

    def __init__(self, name: str, limit: int) -> None:
        self.name = name
        self.limit = max(1, int(limit))
        self._sem = threading.Semaphore(self.limit)
        self._lock = threading.Lock()
        self.active = 0
        self.peak = 0
        self.total = 0
        self.wait_sec = 0.0

    def __enter__(self) -> "StageLimiter":
        t0 = time.perf_counter()
        self._sem.acquire()
        waited = time.perf_counter() - t0
        with self._lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.total += 1
            self.wait_sec += waited
        return self

    def __exit__(self, *exc: object) -> None:
        with self._lock:
            self.active -= 1
        self._sem.release()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "stage": self.name,
                "limit": self.limit,
                "peak_concurrent": self.peak,
                "tasks": self.total,
                "total_wait_sec": round(self.wait_sec, 2),
            }


class ParallelRunner:
    """把「每首歌一个任务」分派到线程池，并在任务内部用分级限制器约束各阶段。"""

    def __init__(self, cfg: Config, file_workers: int | None = None,
                 sep_workers: int | None = None, tr_workers: int | None = None) -> None:
        pool_cfg = cfg.section("pool")
        cpu = _cpu_count()

        # 文件级并发：分离阶段才是瓶颈，因此不需要开满核心
        self.file_workers = int(file_workers or pool_cfg.get("max_workers") or 0) or max(1, min(4, cpu // 4))
        # 分离并发：受显存/内存约束
        self.sep_workers = int(sep_workers or cfg.get("separator.max_concurrent", 1)) or 1
        # 转录并发：ONNX 推理在 C 层执行并释放 GIL，可以开到接近核心数
        self.tr_workers = int(tr_workers or 0) or max(1, min(8, max(2, cpu - 2)))

        self.sep_limiter = StageLimiter("separate", self.sep_workers)
        self.tr_limiter = StageLimiter("transcribe", self.tr_workers)
        self._cancel = threading.Event()

    # ---------- 执行 ----------

    def run(self, tasks: Iterable[tuple[str, Callable[..., Any]]],
            on_event: Callable[[str, dict[str, Any]], None] | None = None
            ) -> list[JobResult]:
        """tasks 为 (job_id, 可调用对象) 序列。

        可调用对象签名为 fn(job_id, limiter_pair) -> JobResult，
        由调用方（pipeline）决定内部如何分阶段使用限制器。
        """
        task_list = list(tasks)
        results: list[JobResult] = []
        total = len(task_list)

        def emit(event: str, data: dict[str, Any]) -> None:
            if on_event:
                try:
                    on_event(event, data)
                except Exception:
                    pass  # 进度回调失败不应影响主流程

        emit("start", {"total": total, "file_workers": self.file_workers,
                       "sep_workers": self.sep_workers, "tr_workers": self.tr_workers})

        if total == 0:
            return results

        with ThreadPoolExecutor(max_workers=self.file_workers,
                                thread_name_prefix="s2m") as ex:
            futures = {}
            for job_id, fn in task_list:
                if self._cancel.is_set():
                    break
                futures[ex.submit(fn, job_id, (self.sep_limiter, self.tr_limiter))] = job_id

            done = 0
            for fut in as_completed(futures):
                job_id = futures[fut]
                done += 1
                try:
                    res = fut.result()
                    if not isinstance(res, JobResult):
                        res = JobResult(job_id=job_id, source="?", ok=True, payload={"raw": res})
                except Exception as e:
                    res = JobResult(job_id=job_id, source="?", ok=False,
                                    error=f"{e.__class__.__name__}: {e}")
                results.append(res)
                emit("job_done", {"job_id": job_id, "done": done, "total": total,
                                  "result": res.to_dict()})

        emit("finish", {"total": total, "succeeded": sum(1 for r in results if r.ok),
                        "limiters": self.stats()})
        return results

    # ---------- 控制 ----------

    def cancel(self) -> None:
        """请求取消。已提交的任务会自行在阶段边界检查并尽早返回。"""
        self._cancel.set()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def stats(self) -> dict[str, Any]:
        return {
            "file_workers": self.file_workers,
            "stages": [self.sep_limiter.snapshot(), self.tr_limiter.snapshot()],
            "cpu_count": _cpu_count(),
        }


def _cpu_count() -> int:
    import os

    try:
        return len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except AttributeError:
        return os.cpu_count() or 4
