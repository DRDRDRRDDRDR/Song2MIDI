"""本地 Web 服务。

设计取向：默认走「本地路径」而不是上传。原因很实际 —— 待处理的是几十兆的
无损音频，上传到本机服务再落盘等于白拷一遍，纯属浪费磁盘和时间。
上传通道仍然保留，用于处理来自浏览器或其他设备的文件。

任务在后台线程中执行，前端轮询进度。之所以不做 WebSocket：处理一首歌要几
分钟，秒级轮询已经足够，而且断线重连后状态不会丢。
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import RESOURCE_ROOT, Config, load_config
from .ingest import SUPPORTED_EXTS
from .pipeline import PipelineOptions, Song2MidiPipeline
from .pool import ParallelRunner

# FastAPI 通过函数的 __globals__ 解析注解类型。本模块启用了
# `from __future__ import annotations`，注解在运行期是字符串，若这些名字只在
# create_app() 内部导入，注解就解析不到 —— FastAPI 会把 request 当成查询参数，
# 返回 422 "loc: [query, request]"。因此必须全部提到模块作用域。
try:
    from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles

    FASTAPI_AVAILABLE = True
except ImportError:  # pragma: no cover
    FASTAPI_AVAILABLE = False
    FastAPI = File = HTTPException = Query = Request = UploadFile = None  # type: ignore
    FileResponse = HTMLResponse = JSONResponse = StaticFiles = None  # type: ignore

__all__ = ["create_app", "JobManager", "JobRecord"]


def _jsonable(node):
    """递归把 numpy 标量等不可序列化类型转成原生 Python 类型。

    Web 层是序列化边界，任何一处 numpy 残留都会让整个任务查询接口返回 500
    （实测 pretty_midi 的 instrument.program 就是 np.int64，一个字段炸掉
    了 /api/jobs 与 /api/jobs/{id} 两个接口）。
    根因已在 export.py 里单独修掉，这里再做一道兜底 ——
    任务查询是前端轮询的核心接口，不允许因为单个字段的类型问题整体失效。
    """
    if isinstance(node, dict):
        return {str(k): _jsonable(v) for k, v in node.items()}
    if isinstance(node, (list, tuple)):
        return [_jsonable(v) for v in node]
    try:
        import numpy as np

        if isinstance(node, np.generic):
            return node.item()
        if isinstance(node, np.ndarray):
            return _jsonable(node.tolist())
    except ImportError:
        pass
    return node


@dataclass
class JobRecord:
    job_id: str
    source: str
    status: str = "pending"      # pending | running | done | failed | cancelled
    stage: str = ""
    stage_history: list[dict[str, Any]] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    summary: dict[str, Any] = field(default_factory=dict)
    export: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    stem_audio: dict[str, str] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)

    @property
    def elapsed(self) -> float:
        return (self.finished_at or time.time()) - self.started_at

    def to_dict(self) -> dict[str, Any]:
        return _jsonable({
            "job_id": self.job_id,
            "source": self.source,
            "filename": Path(self.source).name,
            "status": self.status,
            "stage": self.stage,
            "elapsed_sec": round(self.elapsed, 1),
            "duration_sec": self.summary.get("duration_sec"),
            "tempo": self.summary.get("tempo"),
            "stems": self.summary.get("stems", []),
            "timings": self.summary.get("timings", {}),
            "warnings": self.summary.get("warnings", []),
            "separator": self.summary.get("separator"),
            "export": self.export,
            "stem_audio": self.stem_audio,
            "error": self.error,
            "options": self.options,
            "stage_history": self.stage_history[-40:],
        })


class JobManager:
    """任务登记与并发调度。整个 Web 服务生命周期内只有一个实例。"""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.jobs: dict[str, JobRecord] = {}
        self._lock = threading.Lock()
        self._runner: ParallelRunner | None = None
        self._thread: threading.Thread | None = None
        self._pipeline: Song2MidiPipeline | None = None

    # ---------- 查询 ----------

    def get(self, job_id: str) -> JobRecord | None:
        return self.jobs.get(job_id)

    def list(self) -> list[JobRecord]:
        return sorted(self.jobs.values(), key=lambda j: j.started_at, reverse=True)

    def clear_finished(self) -> int:
        with self._lock:
            gone = [k for k, v in self.jobs.items()
                    if v.status in ("done", "failed", "cancelled")]
            for k in gone:
                self.jobs.pop(k, None)
        return len(gone)

    # ---------- 提交 ----------

    def submit(self, sources: list[str], options: dict[str, Any] | None = None) -> list[str]:
        options = options or {}
        records: list[JobRecord] = []
        for src in sources:
            p = Path(src)
            if not p.is_file():
                continue
            rec = JobRecord(job_id=uuid.uuid4().hex[:12], source=str(p.resolve()),
                            options=options)
            self.jobs[rec.job_id] = rec
            records.append(rec)

        if not records:
            return []

        self._thread = threading.Thread(
            target=self._run_batch, args=(records, options), daemon=True,
            name="s2m-batch",
        )
        self._thread.start()
        return [r.job_id for r in records]

    def _build_options(self, opts: dict[str, Any]) -> PipelineOptions:
        stems = opts.get("stems") or ["vocals", "drums", "bass", "other"]
        if isinstance(stems, str):
            stems = [s.strip() for s in stems.split(",") if s.strip()]
        # separate 三态：None=自动检测，True=强制分离，False=跳过分离
        sep = opts.get("separate", None)
        if sep in (None, "auto", ""):
            separate = None
        else:
            separate = bool(sep)
        return PipelineOptions(
            separate=separate,
            transcribe_drums=bool(opts.get("transcribe_drums", True)),
            transcribe_chords=bool(opts.get("transcribe_chords", True)),
            stems=tuple(stems),
            start=opts.get("start"),
            duration=opts.get("duration"),
            chord_source=str(opts.get("chord_source", "other")),
            quantize=False if opts.get("no_quantize") else None,
        )

    def _run_batch(self, records: list[JobRecord], opts: dict[str, Any]) -> None:
        # 分离引擎档位可在单次任务中覆盖
        cfg = self.cfg
        if opts.get("model"):
            cfg._data.setdefault("separator", {})["model"] = opts["model"]
        if opts.get("device"):
            cfg._data.setdefault("separator", {})["device"] = opts["device"]

        pipeline = Song2MidiPipeline(cfg)
        self._pipeline = pipeline
        p_opts = self._build_options(opts)
        runner = ParallelRunner(cfg, file_workers=opts.get("jobs"))
        self._runner = runner

        by_id = {r.job_id: r for r in records}

        def task_for(rec: JobRecord):
            def task(job_id: str, limiters):
                rec.status = "running"
                last_stage = {"name": ""}

                def on_event(event: str, data: dict) -> None:
                    if event == "stage":
                        name = data.get("stage", "")
                        rec.stage = name
                        last_stage["name"] = name
                        rec.stage_history.append(
                            {"stage": name, "at": round(time.time() - rec.started_at, 2)})
                    elif event == "separate_done":
                        rec.stage_history.append(
                            {"stage": "separate_done", "seconds": data.get("seconds"),
                             "at": round(time.time() - rec.started_at, 2)})
                    elif event == "stem_done":
                        rec.stage_history.append(
                            {"stage": f"stem:{data.get('stem')}", "notes": data.get("notes"),
                             "seconds": data.get("seconds"),
                             "at": round(time.time() - rec.started_at, 2)})

                from .pool import JobResult

                t0 = time.perf_counter()
                res = JobResult(job_id=job_id, source=rec.source)
                try:
                    stems, summary = pipeline.process(rec.source, p_opts, limiters, on_event)
                    rec.summary = summary
                    rec.stem_audio = self._persist_stems(summary, pipeline, p_opts)
                    rec.export = pipeline.export(stems, summary)
                    res.duration_sec = summary.get("duration_sec", 0.0)
                    res.stage_timings = summary.get("timings", {})
                    res.payload = {"summary": summary, "export": rec.export}
                    res.ok = True
                    rec.status = "done"
                except Exception as e:
                    import traceback

                    rec.status = "failed"
                    rec.error = f"{e.__class__.__name__}: {e}"
                    rec.stage_history.append({"stage": "error", "error": rec.error,
                                              "at": round(time.time() - rec.started_at, 2)})
                    res.error = rec.error + "\n" + traceback.format_exc()[-1200:]
                finally:
                    res.elapsed_sec = time.perf_counter() - t0
                    rec.finished_at = time.time()
                return res

            return task

        try:
            runner.run([(r.job_id, task_for(r)) for r in records])
        except Exception as e:
            for r in records:
                if r.status not in ("done", "failed"):
                    r.status = "failed"
                    r.error = f"批处理异常: {e}"
                    r.finished_at = time.time()

    def _persist_stems(self, summary: dict, pipeline: Song2MidiPipeline,
                       opts: PipelineOptions) -> dict[str, str]:
        """把分离出的分轨音频落到 work 目录，供前端试听。

        保留分轨音频而不是处理完就删，是为了让用户能听出「到底是分离错了
        还是转录错了」—— 这是排查音质问题时唯一有效的办法。
        """
        base = Path(summary["source"]).stem
        dst_dir = self.cfg.path("paths.work_dir") / "stems" / base
        mapping: dict[str, str] = {}
        for wav in sorted(dst_dir.glob("*.wav")):
            mapping[wav.stem] = wav.name
        return mapping

    # ---------- 取消 ----------

    def cancel(self, job_id: str | None = None) -> bool:
        if self._runner is None:
            return False
        if job_id:
            rec = self.jobs.get(job_id)
            if rec and rec.status in ("pending", "running"):
                rec.status = "cancelled"
                rec.finished_at = time.time()
        else:
            for r in self.jobs.values():
                if r.status in ("pending", "running"):
                    r.status = "cancelled"
                    r.finished_at = time.time()
        self._runner.cancel()
        return True


def create_app(cfg: Config | None = None):
    """构建 FastAPI 应用。"""
    if not FASTAPI_AVAILABLE:
        raise RuntimeError(
            "缺少 fastapi 依赖，无法启动 Web 服务。\n"
            "请先安装：pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/"
        )

    cfg = cfg or load_config()
    manager = JobManager(cfg)
    static_dir = RESOURCE_ROOT / "app" / "static"

    app = FastAPI(title="Song2MIDI", version="0.1.0")

    # ---------- 页面 ----------

    @app.get("/", response_class=HTMLResponse)
    def index():
        f = static_dir / "index.html"
        if not f.is_file():
            return HTMLResponse("<h1>前端文件缺失</h1>", status_code=500)
        return HTMLResponse(f.read_text(encoding="utf-8"))

    if static_dir.is_dir():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # ---------- 环境与配置 ----------

    @app.get("/api/env")
    def env():
        import importlib

        def has(name: str) -> str | None:
            try:
                m = importlib.import_module(name)
                return getattr(m, "__version__", "installed")
            except Exception:
                return None

        ff = None
        ff_error = None
        try:
            from .ffmpeg_tools import get_tools

            t = get_tools()
            ff = {"ffmpeg": t.ffmpeg, "ffprobe": t.ffprobe, "version": t.version}
        except Exception as e:
            ff_error = str(e)

        models = RESOURCE_ROOT / "models" / "nmp.onnx"
        return {
            "ffmpeg": ff,
            "ffmpeg_error": ff_error,
            "onnx_model": {"path": str(models), "exists": models.is_file(),
                           "bytes": models.stat().st_size if models.is_file() else 0},
            "modules": {n: has(n) for n in
                        ["numpy", "scipy", "soundfile", "librosa", "onnxruntime",
                         "pretty_midi", "torch", "torchaudio", "demucs", "fastapi"]},
            "config": {
                "out_dir": str(cfg.path("paths.out_dir")),
                "work_dir": str(cfg.path("paths.work_dir")),
                "separator_model": cfg.get("separator.model"),
                "device": cfg.get("separator.device"),
                "quantize": cfg.get("postprocess.quantize"),
                "bpm": cfg.get("postprocess.bpm"),
            },
        }

    # ---------- 文件浏览 ----------

    @app.get("/api/browse")
    def browse(path: str = Query(default="")):
        """列出某个目录下的音频文件，供前端做本地文件选择。"""
        roots = [Path.home() / "Downloads", Path.home() / "Desktop", Path.home() / "Music"]
        target = Path(path) if path else None
        if target is None or not target.is_dir():
            return {"path": "", "roots": [str(r) for r in roots if r.is_dir()],
                    "dirs": [], "files": []}
        dirs, files = [], []
        try:
            for item in sorted(target.iterdir()):
                if item.name.startswith("."):
                    continue
                if item.is_dir():
                    dirs.append({"name": item.name, "path": str(item)})
                elif item.suffix.lower() in SUPPORTED_EXTS:
                    files.append({"name": item.name, "path": str(item),
                                  "mb": round(item.stat().st_size / 1048576, 2)})
        except (OSError, PermissionError) as e:
            raise HTTPException(400, f"无法读取目录: {e}")
        return {"path": str(target), "parent": str(target.parent),
                "roots": [str(r) for r in roots if r.is_dir()],
                "dirs": dirs[:200], "files": files[:500]}

    # ---------- 任务 ----------

    @app.post("/api/jobs")
    async def create_jobs(request: Request):
        body = await request.json()
        paths = body.get("paths") or []
        if not paths:
            raise HTTPException(400, "paths 不能为空")
        options = body.get("options") or {}
        ids = manager.submit(paths, options)
        if not ids:
            raise HTTPException(400, "没有有效的文件路径")
        return {"job_ids": ids}

    @app.post("/api/upload")
    async def upload(file: UploadFile = File(...)):
        """接收上传的音频，落到 work/uploads/ 后返回本地路径。

        上传通道只用于浏览器或其他设备送来的文件；本机已有的文件应当
        直接用 /api/browse 选路径，避免几十兆音频白白拷贝一遍。
        """
        name = Path(file.filename or "upload.bin").name
        ext = Path(name).suffix.lower()
        if ext not in SUPPORTED_EXTS:
            raise HTTPException(400, f"不支持的格式: {ext}")
        dst_dir = cfg.path("paths.work_dir") / "uploads"
        dst_dir.mkdir(parents=True, exist_ok=True)
        # 加时间戳前缀，避免同名文件互相覆盖
        dst = dst_dir / f"{int(time.time())}_{name}"
        size = 0
        with dst.open("wb") as fh:
            while chunk := await file.read(1 << 20):
                fh.write(chunk)
                size += len(chunk)
        if size == 0:
            dst.unlink(missing_ok=True)
            raise HTTPException(400, "上传内容为空")
        return {"path": str(dst), "bytes": size}

    @app.post("/api/jobs/cancel-all")
    def cancel_all():
        manager.cancel(None)
        return {"ok": True}

    @app.get("/api/jobs")
    def list_jobs():
        return {"jobs": [j.to_dict() for j in manager.list()]}

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        rec = manager.get(job_id)
        if not rec:
            raise HTTPException(404, "任务不存在")
        return rec.to_dict()

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str):
        if not manager.get(job_id):
            raise HTTPException(404, "任务不存在")
        manager.cancel(job_id)
        return {"ok": True}

    @app.delete("/api/jobs")
    def clear_jobs():
        return {"removed": manager.clear_finished()}

    # ---------- 产物下载 ----------

    @app.get("/api/jobs/{job_id}/audio/{stem}")
    def job_audio(job_id: str, stem: str):
        rec = manager.get(job_id)
        if not rec:
            raise HTTPException(404, "任务不存在")
        base = Path(rec.source).stem
        p = cfg.path("paths.work_dir") / "stems" / base / f"{stem}.wav"
        if not p.is_file():
            raise HTTPException(404, f"分轨音频不存在: {stem}")
        return FileResponse(str(p), media_type="audio/wav", filename=p.name)

    @app.get("/api/jobs/{job_id}/midi/{filename}")
    def job_midi(job_id: str, filename: str):
        rec = manager.get(job_id)
        if not rec:
            raise HTTPException(404, "任务不存在")
        # 防目录穿越：只允许取 out 目录下的直接子文件
        safe = Path(filename).name
        p = cfg.path("paths.out_dir") / safe
        if not p.is_file():
            raise HTTPException(404, f"文件不存在: {safe}")
        media = "audio/midi" if p.suffix.lower() == ".mid" else "application/json"
        return FileResponse(str(p), media_type=media, filename=safe)

    @app.get("/api/jobs/{job_id}/source")
    def job_source(job_id: str):
        rec = manager.get(job_id)
        if not rec or not Path(rec.source).is_file():
            raise HTTPException(404, "源文件不存在")
        return FileResponse(rec.source, filename=Path(rec.source).name)

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):
        return JSONResponse(status_code=500,
                            content={"error": f"{exc.__class__.__name__}: {exc}"})

    return app


def serve() -> int:
    """启动本地服务。"""
    import uvicorn

    cfg = load_config()
    host = str(cfg.get("server.host", "127.0.0.1"))
    port = int(cfg.get("server.port", 8756))
    app = create_app(cfg)

    url = f"http://{host}:{port}"
    print("=" * 64)
    print("Song2MIDI 本地服务")
    print("=" * 64)
    print(f"访问地址: {url}")
    print(f"输出目录: {cfg.path('paths.out_dir')}")
    print("按 Ctrl+C 停止")
    print("=" * 64)

    if cfg.get("server.open_browser", True):
        def _open():
            import time as _t
            import webbrowser

            _t.sleep(1.5)
            try:
                webbrowser.open(url)
            except Exception:
                pass

        threading.Thread(target=_open, daemon=True).start()

    uvicorn.run(app, host=host, port=port, log_level="info")
    return 0
