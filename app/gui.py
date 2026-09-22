"""桌面界面（PySide6）。

设计取舍：
    不用 tkinter —— 隔离环境的 Python 3.13 未附带 tkinter（连 DLL 都没有），
    补装不可行（扩展模块 ABI 与版本绑定）。PySide6 是纯 wheel 依赖，
    在 3.13 上直接可用，且 PyInstaller 支持成熟。

    处理在独立线程中跑，主线程只负责界面刷新。分离一首歌可能耗时数分钟，
    若在工作线程外同步执行，界面会整段卡死。取消采用「协作式中断」：
    设置标志位，任务在文件边界与阶段边界检查后尽早退出，而不是强杀线程。
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QDesktopServices, QFont, QGuiApplication
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFrame,
    QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMainWindow, QMessageBox, QProgressBar, QPushButton,
    QPlainTextEdit, QScrollArea, QSizePolicy, QSpinBox, QSplitter, QTabWidget,
    QVBoxLayout, QWidget,
)

__all__ = ["run_gui"]

STEAM_CN = {
    "vocals": "人声", "drums": "鼓", "bass": "贝斯",
    "other": "其他", "chords": "和弦", "mix": "混音",
}


# ---------------------------------------------------------------- 工作线程

class _Worker(QObject):
    """在后台线程里跑处理流程，通过信号把进度送回主线程。"""

    log = Signal(str, str)          # (级别, 文本)
    stage = Signal(str, str)        # (文件名, 当前阶段)
    progress = Signal(int)          # 0~100 的进度
    file_done = Signal(dict)        # 单个文件的结果
    all_done = Signal(list, float)  # (结果列表, 总耗时)

    def __init__(self, cfg, paths: list[str], options: dict) -> None:
        super().__init__()
        self.cfg = cfg
        self.paths = paths
        self.options = options
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    # ---------- 执行 ----------

    def run(self) -> None:
        t0 = time.time()
        from .pipeline import PipelineOptions, Song2MidiPipeline
        from .pool import ParallelRunner

        opts = PipelineOptions(
            separate=self.options.get("separate"),
            transcribe_drums=bool(self.options.get("drums", True)),
            transcribe_chords=bool(self.options.get("chords", True)),
            stems=tuple(self.options.get("stems") or ("vocals", "drums", "bass", "other")),
            start=self.options.get("start"),
            duration=self.options.get("duration"),
            quantize=False if self.options.get("no_quantize") else None,
        )

        # 引擎与档位写回配置
        self.cfg._data.setdefault("separator", {})["engine"] = \
            self.options.get("engine", "demucs")
        if self.options.get("engine") == "roformer":
            self.cfg._data["separator"]["roformer_type"] = \
                self.options.get("roformer_type", "bs_roformer")
        else:
            self.cfg._data["separator"]["model"] = self.options.get("model", "htdemucs")
        if self.options.get("device"):
            self.cfg._data["separator"]["device"] = self.options["device"]
        # 降噪与清理：写回配置。三项都允许「关闭」，因此不能用 truthy 判断
        # （0 和 False 都是合法值，写成 if x: 会把它们当成未设置而跳过）。
        hp = self.options.get("highpass_hz")
        if hp is not None:
            self.cfg._data.setdefault("preprocess", {})["highpass_hz"] = float(hp)

        mc = self.options.get("midi_clean")
        if mc is not None:
            cl = self.cfg._data.setdefault("postprocess", {}).setdefault("cleanup", {})
            if mc == "off":
                cl["enabled"] = False
            elif mc == "light":
                cl.update({"enabled": True, "drop_shorter_than_ms": 25,
                           "drop_weak_below": 0, "weak_max_ms": 0,
                           "dedupe_repeat_ms": 0, "max_polyphony": 0})
            else:      # medium
                cl.update({"enabled": True, "drop_shorter_than_ms": 30,
                           "drop_weak_below": 35, "weak_max_ms": 80,
                           "dedupe_repeat_ms": 50, "max_polyphony": 16})

        cd = self.options.get("chord_density")
        if cd is not None:
            ch = self.cfg._data.setdefault("chords", {})
            if cd == "off":
                ch.update({"min_duration_ms": 250, "merge_short_ms": 0})
            elif cd == "moderate":
                ch.update({"min_duration_ms": 400, "merge_short_ms": 300})
            else:      # strong
                ch.update({"min_duration_ms": 600, "merge_short_ms": 400})

        if self.options.get("pitch_engine"):
            self.cfg._data.setdefault("transcription", {})["engine"] = \
                self.options["pitch_engine"]
        if self.options.get("quantize"):
            self.cfg._data.setdefault("postprocess", {})["quantize"] = self.options["quantize"]
        if self.options.get("bpm") is not None:
            self.cfg._data.setdefault("postprocess", {})["bpm"] = self.options["bpm"]

        self.log.emit("info", f"开始处理 {len(self.paths)} 个文件")
        self.log.emit("info", f"输出目录：{self.cfg.path('paths.out_dir')}")

        pipeline = Song2MidiPipeline(self.cfg)
        runner = ParallelRunner(self.cfg,
                                file_workers=self.options.get("jobs"),
                                sep_workers=self.options.get("sep_jobs"))

        sep_desc = "已关闭（整首单轨转录）"
        if opts.separate:
            try:
                sep = pipeline.separator
                sep_desc = f"{getattr(sep, 'model_name', '?')} @ {getattr(sep, 'device', '?')}"
            except Exception as e:
                self.log.emit("error", f"分离引擎加载失败：{e}")
                self.all_done.emit([], time.time() - t0)
                return
        self.log.emit("info", f"分离引擎：{sep_desc}")
        self.log.emit("info", f"并行：文件级 {runner.file_workers} / "
                              f"分离 {runner.sep_workers} / 转录 {runner.tr_workers}")

        def make_task(src: str):
            def task(job_id: str, limiters):
                from .pool import JobResult

                name = Path(src).name
                res = JobResult(job_id=job_id, source=src)
                t1 = time.time()
                if self._cancelled:
                    res.error = "已取消"
                    res.elapsed_sec = 0.0
                    return res

                def on_event(event: str, data: dict) -> None:
                    if event == "stage":
                        self.stage.emit(name, data.get("stage", ""))
                        # 单文件时进度条按阶段推进，避免「一直 0 然后突然 100」
                        if len(self.paths) == 1:
                            st = data.get("stage", "")
                            frac = {"ingest": 8, "solo_detect": 15,
                                    "separate": 40, "postprocess": 88}.get(st)
                            if frac is None and st.startswith("transcribe:"):
                                frac = 68
                            if frac is not None:
                                self.progress.emit(max(getattr(self, "_last_prog", 0), frac))
                                self._last_prog = max(getattr(self, "_last_prog", 0), frac)
                    elif event == "solo_detect_done":
                        if data.get("solo"):
                            self.log.emit("info", f"[{name}] 自动检测："
                                          f"{data.get('reason')} → 跳过分离，整首直转")
                        else:
                            self.log.emit("info", f"[{name}] 自动检测："
                                          f"{data.get('reason')} → 执行分离")
                    elif event == "separate_done":
                        self.log.emit("ok", f"[{name}] 分离完成，耗时 {data.get('seconds')}s")
                    elif event == "stem_done":
                        self.log.emit("info",
                                      f"[{name}] {STEAM_CN.get(data.get('stem'), data.get('stem'))} "
                                      f"转录完成：{data.get('notes')} 个音符 "
                                      f"({data.get('seconds')}s)")

                try:
                    stems, summary = pipeline.process(src, opts, limiters, on_event)
                    export = pipeline.export(stems, summary)
                    res.payload = {"summary": summary, "export": export}
                    res.duration_sec = summary.get("duration_sec", 0.0)
                    res.stage_timings = summary.get("timings", {})
                    res.ok = True
                except Exception as e:
                    res.error = f"{e.__class__.__name__}: {e}"
                    self.log.emit("error", f"[{name}] 失败：{res.error}")
                    self.log.emit("debug", traceback.format_exc()[-1200:])
                finally:
                    res.elapsed_sec = time.time() - t1
                # 必须在 finally 之后 emit，否则 elapsed_sec 还没赋值，
                # 结果视图里会显示「处理耗时 0.0 秒」
                self.file_done.emit(res.to_dict())
                return res

            return task

        tasks = [(Path(p).stem, make_task(p)) for p in self.paths]
        results = runner.run(tasks, on_event=lambda ev, d: self._on_runner_event(ev, d))

        if self._cancelled:
            self.log.emit("warn", "任务已取消")

        # runner.run 返回真正的 JobResult 列表，这里转成 dict 再发出去。
        # 之前漏接返回值、发了个永远为空的列表，导致界面显示「成功 0 / 共 0」。
        self.all_done.emit([r.to_dict() for r in results], time.time() - t0)

    def _on_runner_event(self, event: str, data: dict) -> None:
        if event == "job_done":
            # 多文件：按已完成文件数推进进度（留 5% 给收尾）
            if len(self.paths) > 1:
                self.progress.emit(int(data.get("done", 0) / data.get("total", 1) * 95))
            r = data.get("result", {})
            name = Path(r.get("source", "")).name
            ok = r.get("ok")
            sp = r.get("speedup_vs_realtime")
            tail = f"（{sp:.1f}x 实时）" if ok and sp else ""
            self.log.emit("ok" if ok else "error",
                          f"[{data.get('done')}/{data.get('total')}] "
                          f"{'完成' if ok else '失败'} {name} "
                          f"耗时 {r.get('elapsed_sec', 0):.1f}s{tail}")


# ---------------------------------------------------------------- 主窗口

def _decode_auto(raw: bytes) -> str:
    """在 UTF-8 与 GBK 之间挑「解码后中文更多」的那个。

    为什么不简单地「先试 UTF-8，报错再 GBK」：GBK 的双字节序列**常常恰好也是
    合法的 UTF-8**，于是 UTF-8 解码不会报错、却产出一堆乱码字符（如「�汾」）。
    这里用「落在 CJK 统一表意区的字符数」打分 —— 正确解码时中文是成片的，
    错误解码时几乎一个都匹配不到。
    """
    import re

    if not raw:
        return ""
    cjk = re.compile(r"[\u4e00-\u9fff]")
    best, best_score = None, -1
    for enc in ("utf-8", "gbk"):
        try:
            text = raw.decode(enc)
        except UnicodeDecodeError:
            continue
        # 记分：中文多者胜；出现替换字符要扣分
        score = len(cjk.findall(text)) - text.count("\ufffd") * 5
        if score > best_score:
            best, best_score = text, score
    if best is None:
        return raw.decode("utf-8", "replace")
    return best


class _EnvCheckWorker(QObject):
    """在**独立子进程**里运行环境自检。

    为何不用后台线程：自检要首次 import torch / onnxruntime，既是 CPU 密集、
    又会长时间持有 GIL。与 Qt 主线程同进程时二者互相抢占 —— 实测主线程心跳
    出现 149ms 的超时间隔；用户此时点「添加文件」，Windows 原生文件对话框的
    shell 初始化被拖慢，窗口就被系统标记为「未响应」。

    放子进程把这份开销完全隔离：实测心跳最大 92ms、0 次卡顿，
    自检本身也更快（不与主进程已加载的库竞争 DLL 与内存）。

    用 `--check --json` 拿结构化结果，主进程解析后回填状态栏与日志，
    因此不再依赖同进程内的 `tools.check_env`。
    """

    done = Signal(str, int, int, int)   # (完整文本, fail, warn, ok)

    def __init__(self) -> None:
        super().__init__()
        self._results: list = []

    @staticmethod
    def _build_cmd() -> list:
        """冻结环境用 exe 自身，源码环境用 python main.py。"""
        if getattr(sys, "frozen", False):
            return [sys.executable, "--check", "--json"]
        root = Path(__file__).resolve().parent.parent
        return [sys.executable, str(root / "main.py"), "--check", "--json"]

    @staticmethod
    def _render(res: list) -> str:
        """把 JSON 结果重新排版成可读文本，格式与 CLI 自检保持一致。"""
        lines = ["=" * 68, "Song2MIDI 环境自检", "=" * 68]
        for level, title in (("FAIL", "阻塞项"), ("WARN", "警告"), ("OK", "正常")):
            group = [r for r in res if r.get("level") == level]
            if not group:
                continue
            lines.append(f"{'-' * 68}")
            lines.append(f"[{title}]（{len(group)} 项）")
            for r in group:
                lines.append(f"[{level:^4}] {r.get('item', '')}")
                lines.append(f"        {r.get('detail', '')}")
        return "\n".join(lines)

    def run(self) -> None:
        import json as _json
        import subprocess as _sp

        fails = warns = oks = 0
        self._results = []
        text = ""
        try:
            proc = _sp.Popen(
                self._build_cmd(),
                stdout=_sp.PIPE, stderr=_sp.STDOUT,
                creationflags=getattr(_sp, "CREATE_NO_WINDOW", 0),
            )
            out, _ = proc.communicate(timeout=300)
            # 不假定子进程的编码，交给打分式解码挑选
            raw = _decode_auto(out or b"").strip()
            # 有些库会往 stdout 打日志，JSON 之前可能有杂音，从头一个 { 开始截
            i = raw.find("{")
            if i > 0:
                raw = raw[i:]
            data = _json.loads(raw)
            res = list(data.get("results") or [])
            self._results = res
            fails = int(data.get("fail", 0))
            warns = int(data.get("warn", 0))
            oks = sum(1 for r in res if r.get("level") == "OK")
            text = self._render(res)
        except Exception as e:
            text = f"环境自检未能完成：{e.__class__.__name__}: {e}"
            fails = 1
        self.done.emit(text, fails, warns, oks)

    @property
    def results(self) -> list:
        return self._results


_QSS = """
QWidget { background: #1e2021; color: #e6e8ea;
          font-family: "Microsoft YaHei UI","Segoe UI",sans-serif; font-size: 13px; }
QGroupBox { border: 1px solid #34383b; border-radius: 8px; margin-top: 14px;
            padding: 12px 10px 10px 10px; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px;
                   color: #9aa1a7; font-weight: 600; }
QPushButton { background: #2b2f31; border: 1px solid #3c4144; border-radius: 6px;
              padding: 7px 14px; }
QPushButton:hover { background: #343a3d; }
QPushButton:pressed { background: #262a2c; }
QPushButton:disabled { color: #6a7075; background: #26292b; }
QPushButton#primary { background: #2f6fb0; border-color: #2f6fb0; font-weight: 600; }
QPushButton#primary:hover { background: #3a82c9; }
QPushButton#primary:disabled { background: #2a4356; border-color: #2a4356; color: #71818c; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QListWidget, QPlainTextEdit {
    background: #26292b; border: 1px solid #3c4144; border-radius: 6px; padding: 5px 7px;
    selection-background-color: #2f6fb0; }
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus { border-color: #2f6fb0; }
QComboBox::drop-down { border: none; width: 20px; }
QComboBox QAbstractItemView { background: #26292b; border: 1px solid #3c4144;
                              selection-background-color: #2f6fb0; }
QCheckBox { spacing: 7px; }
QCheckBox::indicator { width: 15px; height: 15px; border: 1px solid #4a5054;
                       border-radius: 3px; background: #26292b; }
QCheckBox::indicator:checked { background: #2f6fb0; border-color: #2f6fb0; }
QProgressBar { border: 1px solid #3c4144; border-radius: 6px; background: #26292b;
               text-align: center; height: 18px; }
QProgressBar::chunk { background: #2f6fb0; border-radius: 5px; }
QTabWidget::pane { border: 1px solid #34383b; border-radius: 6px; }
QTabBar::tab { background: #26292b; padding: 6px 16px; border: 1px solid #34383b;
               border-bottom: none; border-top-left-radius: 6px; border-top-right-radius: 6px; }
QTabBar::tab:selected { background: #2f6fb0; }
QListWidget::item { padding: 5px 6px; }
QListWidget::item:selected { background: #2f6fb0; }
QSplitter::handle { background: #2a2d2f; }
QScrollBar:vertical { background: #1e2021; width: 11px; }
QScrollBar::handle:vertical { background: #3c4144; border-radius: 5px; min-height: 24px; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; }
QLabel#hint { color: #8b9298; font-size: 12px; }
QLabel#warn { color: #e0b761; }
"""


class MainWindow(QMainWindow):
    def __init__(self, cfg, initial_files: list[str] | None = None,
                 autorun: bool = False) -> None:
        super().__init__()
        self.cfg = cfg
        self.paths: list[str] = []
        self.worker: _Worker | None = None
        self.thread: QThread | None = None
        self._files_done = 0

        self.setWindowTitle("Song2MIDI · 歌曲转多轨 MIDI")
        # 按屏幕可用区域自适应：左侧控件区堆了 5 个分组，内容最小高度约 980px。
        # 系统缩放 125% 时 1920x1080 的可用高度只有约 824px，写死尺寸必然溢出。
        # 这里取可用区域的 92%，并留出下限，保证任何屏幕都能完整看到窗口。
        scr = QGuiApplication.primaryScreen()
        if scr is not None:
            g = scr.availableGeometry()
            self.resize(max(900, min(1240, int(g.width() * 0.92))),
                        max(560, min(820, int(g.height() * 0.92))))
        else:
            self.resize(1180, 780)
        self.setAcceptDrops(True)

        self._build_ui()
        self._on_pitch_engine_changed()
        self._on_separation_changed()
        self._on_cleanup_changed()
        self._refresh_env()
        # 启动后自动做一次环境自检（在独立子进程里跑，见 _EnvCheckWorker）。
        # 延迟 1.5 秒：先让窗口完全画出来、消息循环跑顺，再启动子进程 ——
        # 子进程创建的瞬间仍会争抢一点 CPU 与磁盘，留出余量可避免
        # 用户刚打开窗口就点「添加文件」时撞上文件对话框的慢启动。
        QTimer.singleShot(1500, self._run_env_check)

        # 从「拖文件到 exe 图标上」启动时的入口：预填文件并自动开始
        if initial_files:
            n = self._add_paths(initial_files)
            self._append_log("ok", f"已从命令行载入 {n} 个文件")
            if autorun and n:
                QTimer.singleShot(400, self._start)

    # ---------------- 界面搭建 ----------------

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(14, 12, 14, 12)
        outer.setSpacing(10)

        outer.addWidget(self._build_header())

        split = QSplitter(Qt.Horizontal)
        split.addWidget(self._build_left())
        split.addWidget(self._build_right())
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([400, 760])
        outer.addWidget(split, 1)

        outer.addWidget(self._build_progress())

    def _build_header(self) -> QWidget:
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        title = QLabel("Song2MIDI")
        f = QFont()
        f.setPointSize(13)
        f.setBold(True)
        title.setFont(f)
        lay.addWidget(title)
        sub = QLabel("歌曲 → 四轨分离 → 多轨 MIDI · 全程本地处理，音频不出本机")
        sub.setObjectName("hint")
        lay.addWidget(sub)
        lay.addStretch(1)
        self.env_label = QLabel("环境检测中…")
        self.env_label.setObjectName("hint")
        lay.addWidget(self.env_label)
        btn = QPushButton("环境自检")
        btn.clicked.connect(self._run_env_check)
        lay.addWidget(btn)
        return w

    def _build_left(self) -> QWidget:
        panel = QWidget()
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 8, 0)
        lay.setSpacing(8)

        # 文件
        gb = QGroupBox("待处理音频")
        gl = QVBoxLayout(gb)
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.file_list.setMinimumHeight(130)
        gl.addWidget(self.file_list)
        row = QHBoxLayout()
        for text, slot in (("添加文件", self._add_files),
                           ("添加文件夹", self._add_folder),
                           ("移除选中", self._remove_selected),
                           ("清空", self._clear_files)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            row.addWidget(b)
        gl.addLayout(row)
        hint = QLabel("也可以把音频文件或文件夹直接拖进窗口")
        hint.setObjectName("hint")
        gl.addWidget(hint)
        lay.addWidget(gb)

        # 输出
        gb2 = QGroupBox("输出目录")
        g2 = QHBoxLayout(gb2)
        self.out_edit = QLineEdit(str(self.cfg.path("paths.out_dir")))
        g2.addWidget(self.out_edit, 1)
        b = QPushButton("选择")
        b.clicked.connect(self._pick_out)
        g2.addWidget(b)
        b2 = QPushButton("打开")
        b2.clicked.connect(lambda: self._open_path(self.out_edit.text()))
        g2.addWidget(b2)
        lay.addWidget(gb2)

        # 分离
        gb3 = QGroupBox("音源分离")
        g3 = QGridLayout(gb3)
        g3.addWidget(QLabel("分离方式"), 0, 0)
        self.cmb_separation = QComboBox()
        self.cmb_separation.addItem("自动检测（推荐）", "auto")
        self.cmb_separation.addItem("强制四轨分离", "force")
        self.cmb_separation.addItem("跳过分离（整首直转）", "skip")
        self.cmb_separation.currentIndexChanged.connect(self._on_separation_changed)
        g3.addWidget(self.cmb_separation, 0, 1)

        self.solo_hint = QLabel("")
        self.solo_hint.setObjectName("hint")
        self.solo_hint.setWordWrap(True)
        g3.addWidget(self.solo_hint, 1, 0, 1, 2)

        g3.addWidget(QLabel("分离引擎"), 2, 0)
        self.cmb_engine = QComboBox()
        self.cmb_engine.addItem("Demucs（快，约 2 分钟/首）", "demucs")
        self.cmb_engine.addItem("BS-RoFormer（质量最好，需 GPU）", "roformer")
        self.cmb_engine.currentIndexChanged.connect(self._on_engine_changed)
        g3.addWidget(self.cmb_engine, 2, 1)

        g3.addWidget(QLabel("权重档位"), 3, 0)
        self.cmb_model = QComboBox()
        g3.addWidget(self.cmb_model, 3, 1)

        g3.addWidget(QLabel("计算设备"), 4, 0)
        self.cmb_device = QComboBox()
        self.cmb_device.addItem("自动", "auto")
        self.cmb_device.addItem("GPU (CUDA)", "cuda")
        self.cmb_device.addItem("CPU", "cpu")
        g3.addWidget(self.cmb_device, 4, 1)

        self.engine_warn = QLabel("")
        self.engine_warn.setObjectName("warn")
        self.engine_warn.setWordWrap(True)
        g3.addWidget(self.engine_warn, 5, 0, 1, 2)
        lay.addWidget(gb3)

        # 转录
        gb4 = QGroupBox("转录与后处理")
        g4 = QGridLayout(gb4)

        g4.addWidget(QLabel("转录引擎"), 0, 0)
        self.cmb_pitch_engine = QComboBox()
        self.cmb_pitch_engine.addItem("自动（推荐）", "auto")
        self.cmb_pitch_engine.addItem("通用 Basic Pitch", "basic_pitch")
        self.cmb_pitch_engine.addItem("钢琴专用（含踏板）", "piano")
        self.cmb_pitch_engine.currentIndexChanged.connect(self._on_pitch_engine_changed)
        g4.addWidget(self.cmb_pitch_engine, 0, 1)

        self.pitch_hint = QLabel("")
        self.pitch_hint.setObjectName("hint")
        self.pitch_hint.setWordWrap(True)
        g4.addWidget(self.pitch_hint, 1, 0, 1, 2)

        self.cb_drums = QCheckBox("鼓组转录")
        self.cb_drums.setChecked(True)
        self.cb_chords = QCheckBox("和弦识别")
        self.cb_chords.setChecked(True)
        g4.addWidget(self.cb_drums, 2, 0)
        g4.addWidget(self.cb_chords, 2, 1)

        g4.addWidget(QLabel("要转录的分轨"), 3, 0, 1, 2)
        stem_w = QWidget()
        sl = QHBoxLayout(stem_w)
        sl.setContentsMargins(0, 0, 0, 0)
        self.stem_boxes: dict[str, QCheckBox] = {}
        for key in ("vocals", "drums", "bass", "other"):
            cb = QCheckBox(STEAM_CN[key])
            cb.setChecked(True)
            self.stem_boxes[key] = cb
            sl.addWidget(cb)
        sl.addStretch(1)
        g4.addWidget(stem_w, 4, 0, 1, 2)

        g4.addWidget(QLabel("量化网格"), 3, 0)
        self.cmb_quant = QComboBox()
        for label, v in (("1/16（推荐）", "1/16"), ("1/8", "1/8"), ("1/4", "1/4"),
                         ("关闭量化", "off")):
            self.cmb_quant.addItem(label, v)
        g4.addWidget(self.cmb_quant, 3, 1)

        g4.addWidget(QLabel("节拍"), 4, 0)
        self.cmb_bpm = QComboBox()
        self.cmb_bpm.addItem("自动估计", "auto")
        self.cmb_bpm.addItem("手动指定", "manual")
        self.cmb_bpm.currentIndexChanged.connect(
            lambda: self.spin_bpm.setEnabled(self.cmb_bpm.currentData() == "manual"))
        g4.addWidget(self.cmb_bpm, 4, 1)
        self.spin_bpm = QDoubleSpinBox()
        self.spin_bpm.setRange(40, 300)
        self.spin_bpm.setValue(120)
        self.spin_bpm.setEnabled(False)
        g4.addWidget(self.spin_bpm, 5, 1)

        g4.addWidget(QLabel("片段（秒）"), 6, 0)
        seg = QWidget()
        segl = QHBoxLayout(seg)
        segl.setContentsMargins(0, 0, 0, 0)
        self.spin_start = QDoubleSpinBox()
        self.spin_start.setRange(0, 100000)
        self.spin_start.setSpecialValueText("从头")
        self.spin_start.setToolTip("起始秒；保持 0 表示从头")
        self.spin_dur = QDoubleSpinBox()
        self.spin_dur.setRange(0, 100000)
        self.spin_dur.setSpecialValueText("整首")
        self.spin_dur.setToolTip("只处理这么长的片段；保持 0 表示整首")
        segl.addWidget(QLabel("起"))
        segl.addWidget(self.spin_start)
        segl.addWidget(QLabel("长"))
        segl.addWidget(self.spin_dur)
        g4.addWidget(seg, 6, 1)
        seg_hint = QLabel("先填片段试跑，确认效果再处理整首，可省下大量等待")
        seg_hint.setObjectName("hint")
        seg_hint.setWordWrap(True)
        g4.addWidget(seg_hint, 7, 0, 1, 2)
        lay.addWidget(gb4)

        # 降噪与清理
        gb5 = QGroupBox("降噪与清理")
        g5 = QGridLayout(gb5)

        g5.addWidget(QLabel("音频高通"), 0, 0)
        self.cmb_highpass = QComboBox()
        self.cmb_highpass.addItem("关闭", 0)
        self.cmb_highpass.addItem("40 Hz（去低频嗡声）", 40)
        self.cmb_highpass.addItem("50 Hz", 50)
        self.cmb_highpass.addItem("60 Hz", 60)
        self.cmb_highpass.currentIndexChanged.connect(self._on_cleanup_changed)
        g5.addWidget(self.cmb_highpass, 0, 1)

        g5.addWidget(QLabel("MIDI 清理"), 1, 0)
        self.cmb_midi_clean = QComboBox()
        self.cmb_midi_clean.addItem("关闭", "off")
        self.cmb_midi_clean.addItem("轻度（去极短音）", "light")
        self.cmb_midi_clean.addItem("中度（+去弱音/重复/限复音）", "medium")
        self.cmb_midi_clean.currentIndexChanged.connect(self._on_cleanup_changed)
        g5.addWidget(self.cmb_midi_clean, 1, 1)

        g5.addWidget(QLabel("和弦降密"), 2, 0)
        self.cmb_chord_density = QComboBox()
        self.cmb_chord_density.addItem("关闭", "off")
        self.cmb_chord_density.addItem("适度（最短 400ms + 归并）", "moderate")
        self.cmb_chord_density.addItem("较强（最短 600ms + 归并）", "strong")
        self.cmb_chord_density.currentIndexChanged.connect(self._on_cleanup_changed)
        g5.addWidget(self.cmb_chord_density, 2, 1)

        self.cleanup_hint = QLabel("")
        self.cleanup_hint.setObjectName("hint")
        self.cleanup_hint.setWordWrap(True)
        g5.addWidget(self.cleanup_hint, 3, 0, 1, 2)
        lay.addWidget(gb5)

        lay.addStretch(1)

        row = QHBoxLayout()
        self.btn_run = QPushButton("开始处理")
        self.btn_run.setObjectName("primary")
        self.btn_run.clicked.connect(self._start)
        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._cancel)
        row.addWidget(self.btn_run, 1)
        row.addWidget(self.btn_cancel)
        lay.addLayout(row)

        self._on_engine_changed()

        # 包一层滚动区：分组越加越多（现在 5 个，内容最小高度约 980px），
        # 小屏或高 DPI 缩放下必然装不下，Qt 会把窗口强行撑高。
        # 有滚动条就不会撑窗口，用户随时能滚到下面的分组。
        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setMinimumWidth(380)
        scroll.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        return scroll

    def _build_right(self) -> QWidget:
        panel = QWidget()
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)

        self.tabs = QTabWidget()
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(4000)
        f = QFont("Consolas" if os.name == "nt" else "monospace")
        f.setPointSize(9)
        self.log_view.setFont(f)
        self.tabs.addTab(self.log_view, "运行日志")

        self.result_view = QPlainTextEdit()
        self.result_view.setReadOnly(True)
        self.result_view.setFont(f)
        self.tabs.addTab(self.result_view, "结果")
        lay.addWidget(self.tabs, 1)

        row = QHBoxLayout()
        b = QPushButton("复制日志")
        b.clicked.connect(self._copy_log)
        row.addWidget(b)
        b2 = QPushButton("打开输出目录")
        b2.clicked.connect(lambda: self._open_path(self.out_edit.text()))
        row.addWidget(b2)
        row.addStretch(1)
        lay.addLayout(row)
        return panel

    def _build_progress(self) -> QWidget:
        w = QFrame()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        lay.addWidget(self.progress, 1)
        self.stage_label = QLabel("就绪")
        self.stage_label.setObjectName("hint")
        self.stage_label.setMinimumWidth(420)
        lay.addWidget(self.stage_label, 2)
        return w

    # ---------------- 环境 ----------------

    def _refresh_env(self) -> None:
        """只做无阻塞的占位显示。

        这里**不能** import torch —— 首次导入实测约 6.5 秒，
        会让窗口迟迟不出现。真正的版本号与 CUDA 状态由后台自检完成后回填
        （见 _on_env_check_done）。
        """
        self._torch_cuda = None          # None = 尚未探测
        self.env_label.setText("环境自检中…")
        self._update_engine_warning()

    def _run_env_check(self) -> None:
        """运行环境自检（后台线程）。

        不走子进程：打包成 exe 后 sys.executable 指向 exe 自身，
        tools/check_env.py 也不再作为独立文件存在，子进程方式必然失败。
        同时必须后台执行 —— 含首次 import torch，主线程会卡住界面。
        """
        th = getattr(self, "_env_thread", None)
        if th is not None and th.isRunning():
            self._append_log("info", "环境自检正在进行中…")
            return

        self.env_label.setText("环境自检中…")
        self._append_log("info", "=" * 56)
        self._append_log("info", "开始环境自检（后台进行，可继续操作界面）")

        self._env_worker = _EnvCheckWorker()
        self._env_thread = QThread(self)
        self._env_worker.moveToThread(self._env_thread)
        self._env_thread.started.connect(self._env_worker.run)
        self._env_worker.done.connect(self._on_env_check_done)
        self._env_worker.done.connect(self._env_thread.quit)
        self._env_thread.start()

    def _on_env_check_done(self, text: str, fails: int, warns: int, oks: int) -> None:
        """自检完成：日志着色输出、状态栏摘要、必要时回填 CUDA 状态。"""
        level_map = {"[FAIL]": "error", "[WARN]": "warn", "[ OK ]": "ok"}
        for line in (text or "").splitlines():
            lv = "info"
            for tag, l in level_map.items():
                if tag in line:
                    lv = l
                    break
            self._append_log(lv, line)

        total = fails + warns + oks
        summary = f"环境自检：{fails} 项阻塞 / {warns} 项警告 / {oks} 项正常"
        self.env_label.setText(summary)
        self._append_log("error" if fails else ("warn" if warns else "ok"), summary)

        # 从自检结果里取出 torch 与 CUDA 的真实状态，回填到状态栏与引擎警告
        try:
            res = self._env_worker.results if self._env_worker else []
            by_item = {r.get("item", ""): r for r in res}
            torch_r = by_item.get("模块 torch")
            cuda_r = by_item.get("CUDA")
            self._torch_cuda = bool(cuda_r and cuda_r.get("level") == "OK")
            bits = []
            if torch_r:
                det = str(torch_r.get("detail", ""))
                bits.append(det.split(" — ")[0].split("（")[0].strip())
            if cuda_r:
                det = str(cuda_r.get("detail", ""))
                bits.append(det.split("，")[0].strip() or "CUDA 可用")
            elif torch_r:
                bits.append("仅 CPU")
            if bits:
                self.env_label.setText(" · ".join(bits) + f"　|　{summary}")
            self._update_engine_warning()
        except Exception as e:
            self._append_log("warn", f"回填环境信息失败：{e.__class__.__name__}: {e}")
            self._torch_cuda = False
            self._update_engine_warning()

        if fails:
            self.tabs.setCurrentIndex(0)
            self._append_log("error", "存在阻塞项，请先按上方提示修复后再处理音频")

    # ---------------- 控件联动 ----------------

    def _separation_value(self):
        """三态：None=自动检测，True=强制分离，False=跳过分离。"""
        v = self.cmb_separation.currentData()
        return None if v == "auto" else (True if v == "force" else False)

    def _on_separation_changed(self) -> None:
        val = self._separation_value()
        # 只有「跳过分离」才禁用引擎/档位/设备选择；
        # 自动检测模式下最终可能还是会分离，因此保持可选。
        enabled = val is not False
        for w in (self.cmb_engine, self.cmb_model, self.cmb_device):
            w.setEnabled(enabled)
        hints = {
            "auto": "自动检测：无鼓、无贝斯的独奏曲（钢琴/吉他/弦乐）会跳过四轨分离、整首直转，"
                    "避免把琴槌瞬态误判成鼓。",
            "force": "始终做四轨分离。注意：纯钢琴/独奏曲会被拆出大量假鼓，不建议。",
            "skip": "跳过分离，整首当一轨转录。适合钢琴独奏、吉他独奏等无节奏组的曲目。",
        }
        self.solo_hint.setText(hints.get(self.cmb_separation.currentData(), ""))
        self._update_engine_warning()

    def _on_cleanup_changed(self) -> None:
        tips = []
        hp = self.cmb_highpass.currentData()
        if hp:
            tips.append(f"高通 {hp}Hz：滤掉低频嗡声，实测对音符内容零损失")
        else:
            tips.append("音频降噪未开启（实测谱门控降噪会削掉钢琴低频，收益为负，故不提供）")
        mc = self.cmb_midi_clean.currentData()
        if mc == "light":
            tips.append("MIDI 清理-轻度：仅去除 <25ms 的极短音（物理上不可能是真实击键）")
        elif mc == "medium":
            tips.append("MIDI 清理-中度：另去低力度短音、同音极短重复，并把同时发声限制在 16")
        if self.cmb_chord_density.currentData() != "off":
            tips.append("和弦降密：提高最短持续时间并归并 A-B-A 中的短段，减少和弦频繁切换")
        self.cleanup_hint.setText("　|　".join(tips))

    def _on_pitch_engine_changed(self) -> None:
        hints = {
            "auto": "自动：检测到无鼓无贝斯的独奏曲时用钢琴专用模型（含踏板检测），"
                    "其余情形用通用模型。",
            "basic_pitch": "始终用通用模型 Basic Pitch。对钢琴曲会把音符切得偏短、且无踏板信息。",
            "piano": "始终用钢琴专用模型（ByteDance 高分辨率钢琴转录，含踏板）。"
                     "对非钢琴曲效果未经验证。",
        }
        self.pitch_hint.setText(hints.get(self.cmb_pitch_engine.currentData(), ""))

    def _on_engine_changed(self) -> None:
        eng = self.cmb_engine.currentData()
        self.cmb_model.clear()
        if eng == "roformer":
            self.cmb_model.addItem("BS-RoFormer 4 轨 · Apache-2.0", "bs_roformer")
        else:
            self.cmb_model.addItem("htdemucs — 最快（推荐）", "htdemucs")
            self.cmb_model.addItem("htdemucs_ft — 质量更好，约 4 倍耗时", "htdemucs_ft")
            self.cmb_model.addItem("hdemucs_mmi — 速度质量居中", "hdemucs_mmi")
            self.cmb_model.addItem("mdx_extra — 质量较高，占用大", "mdx_extra")
        self._update_engine_warning()

    def _update_engine_warning(self) -> None:
        if self._separation_value() is False:
            self.engine_warn.setText("")
            return
        if self.cmb_engine.currentData() != "roformer":
            self.engine_warn.setText("")
            return
        cuda = getattr(self, "_torch_cuda", None)
        if cuda is None:
            self.engine_warn.setText("正在检测 CUDA 状态…")
        elif not cuda:
            self.engine_warn.setText(
                "⚠ 当前 torch 无 CUDA，BS-RoFormer 在 CPU 上实测约 0.05x 实时"
                "（4 分钟的歌需约 79 分钟）。建议改用 Demucs，或换装 CUDA 版 torch。")
        else:
            self.engine_warn.setText("")

    # ---------------- 文件操作 ----------------

    def dragEnterEvent(self, e) -> None:
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e) -> None:
        from .ingest import SUPPORTED_EXTS
        added = 0
        for url in e.mimeData().urls():
            p = Path(url.toLocalFile())
            if p.is_dir():
                added += self._add_paths(sorted(p.rglob("*")))
            elif p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
                added += self._add_paths([p])
        self._log_added(added)

    def _add_paths(self, paths) -> int:
        n = 0
        for p in paths:
            p = Path(p)
            if not p.is_file():
                continue
            s = str(p.resolve())
            if s in self.paths:
                continue
            self.paths.append(s)
            item = QListWidgetItem(p.name)
            item.setToolTip(s)
            self.file_list.addItem(item)
            n += 1
        return n

    def _log_added(self, n: int) -> None:
        if n:
            self._append_log("ok", f"已添加 {n} 个文件，当前共 {len(self.paths)} 个")
        else:
            self._append_log("warn", "没有新增文件（可能重复或格式不支持）")

    def _add_files(self) -> None:
        from .ingest import SUPPORTED_EXTS

        pat = "音频文件 (" + " ".join(f"*{e}" for e in sorted(SUPPORTED_EXTS)) + ");;所有文件 (*.*)"
        files, _ = QFileDialog.getOpenFileNames(self, "选择音频文件", "", pat)
        self._log_added(self._add_paths(files))

    def _add_folder(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "选择文件夹")
        if not d:
            return
        from .ingest import SUPPORTED_EXTS
        self._log_added(self._add_paths(
            sorted(p for p in Path(d).rglob("*")
                   if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS)))

    def _remove_selected(self) -> None:
        for item in self.file_list.selectedItems():
            row = self.file_list.row(item)
            self.file_list.takeItem(row)
            if 0 <= row < len(self.paths):
                self.paths.pop(row)

    def _clear_files(self) -> None:
        self.file_list.clear()
        self.paths.clear()

    def _pick_out(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "选择输出目录", self.out_edit.text())
        if d:
            self.out_edit.setText(d)

    @staticmethod
    def _open_path(p: str) -> None:
        path = Path(p)
        if not path.exists():
            QMessageBox.information(None, "提示", f"路径不存在：{p}")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _copy_log(self) -> None:
        QApplication.clipboard().setText(self.log_view.toPlainText())
        self._append_log("info", "日志已复制到剪贴板")

    def _append_log(self, level: str, text: str) -> None:
        color = {"info": "#c8ccd0", "ok": "#8fd18f", "warn": "#e0b761",
                 "error": "#f08a8a", "debug": "#7d848a"}.get(level, "#c8ccd0")
        stamp = time.strftime("%H:%M:%S")
        for line in (text or "").rstrip().splitlines():
            self.log_view.appendHtml(
                f'<span style="color:#6f767c">{stamp}</span> '
                f'<span style="color:{color}">{self._esc(line)}</span>')
        self.log_view.verticalScrollBar().setValue(
            self.log_view.verticalScrollBar().maximum())

    @staticmethod
    def _esc(s: str) -> str:
        return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))

    # ---------------- 执行 ----------------

    def _collect_options(self) -> dict:
        engine = self.cmb_engine.currentData()
        return {
            "separate": self._separation_value(),
            "engine": engine,
            "pitch_engine": self.cmb_pitch_engine.currentData(),
            "highpass_hz": self.cmb_highpass.currentData(),
            "midi_clean": self.cmb_midi_clean.currentData(),
            "chord_density": self.cmb_chord_density.currentData(),
            "model": self.cmb_model.currentData(),
            # 档位下拉在不同引擎下装的是不同东西，不能无条件取它的值：
            # 引擎为 demucs 时那个值是 htdemucs，若直接当作 roformer_type 传下去
            # 会污染配置。因此只在 roformer 引擎下取，否则用固定默认值。
            "roformer_type": (self.cmb_model.currentData()
                              if engine == "roformer" else "bs_roformer"),
            "device": self.cmb_device.currentData(),
            "drums": self.cb_drums.isChecked(),
            "chords": self.cb_chords.isChecked(),
            "stems": [k for k, cb in self.stem_boxes.items() if cb.isChecked()],
            "no_quantize": self.cmb_quant.currentData() == "off",
            "quantize": self.cmb_quant.currentData(),
            "bpm": (self.spin_bpm.value()
                    if self.cmb_bpm.currentData() == "manual" else "auto"),
            "start": self.spin_start.value() or None,
            "duration": self.spin_dur.value() or None,
        }

    def _start(self) -> None:
        if not self.paths:
            QMessageBox.information(self, "提示", "请先添加要处理的音频文件。")
            return
        if self.thread and self.thread.isRunning():
            return

        opts = self._collect_options()
        if not opts["stems"] and opts["separate"] is not False:
            QMessageBox.information(self, "提示", "至少选择一个要转录的分轨。")
            return

        # 输出目录写回配置
        self.cfg._data.setdefault("paths", {})["out_dir"] = self.out_edit.text()

        self._files_done = 0
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.result_view.clear()
        self.tabs.setCurrentIndex(0)

        self.worker = _Worker(self.cfg, list(self.paths), opts)
        self.thread = QThread(self)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.log.connect(self._append_log)
        self.worker.stage.connect(self._on_stage)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.file_done.connect(self._on_file_done)
        self.worker.all_done.connect(self._on_all_done)
        self.thread.start()

        self.btn_run.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self._append_log("info", "=" * 60)

    def _cancel(self) -> None:
        if self.worker:
            self.worker.cancel()
            self._append_log("warn", "已请求取消，将在当前阶段结束后停止…")

    def _on_stage(self, name: str, stage: str) -> None:
        cn = {"ingest": "解码", "separate": "音源分离", "postprocess": "后处理"}
        label = cn.get(stage, stage)
        if stage.startswith("transcribe:"):
            label = f"转录 {STEAM_CN.get(stage.split(':', 1)[1], stage.split(':', 1)[1])}"
        self.stage_label.setText(f"{name} · {label}")

    def _on_file_done(self, res: dict) -> None:
        self._files_done += 1
        total = max(len(self.paths), 1)
        self.progress.setValue(int(self._files_done / total * 95))

        if not res.get("ok"):
            self.result_view.appendPlainText(
                f"✗ {Path(res.get('source', '')).name}\n    错误：{res.get('error')}\n")
            return

        s = res.get("payload", {}).get("summary", {})
        e = res.get("payload", {}).get("export", {})
        lines = [
            f"■ {Path(res['source']).name}",
            f"    音频 {res.get('duration_sec', 0):.0f} 秒 · 处理耗时 {res.get('elapsed_sec', 0):.1f} 秒",
        ]
        t = s.get("tempo") or {}
        if t:
            lines.append(f"    节拍 {t.get('bpm')} BPM（相位 {t.get('phase_sec')}s，"
                         f"置信度 {t.get('confidence')}）")
        sep = s.get("separator")
        if sep:
            lines.append(f"    分离：{sep.get('model')} @ {sep.get('device')}")
        for st in s.get("stems", []):
            nm = STEAM_CN.get(st.get("stem"), st.get("stem"))
            lines.append(f"    {nm:<6} {st.get('count', 0):>5} 音符   "
                         f"音域 {st.get('pitch_min_name', '-')}–{st.get('pitch_max_name', '-')}   "
                         f"平均时值 {st.get('duration_mean_ms', '-')} ms")
        for f in e.get("midi_files", []):
            lines.append(f"    → {f}")
        for w in s.get("warnings") or []:
            lines.append(f"    ! {w}")
        lines.append("")
        self.result_view.appendPlainText("\n".join(lines))
        self.tabs.setCurrentIndex(1)

    def _on_all_done(self, results: list, total_sec: float) -> None:
        if self.thread:
            self.thread.quit()
            self.thread.wait(3000)
        self.thread = None
        self.worker = None
        self.btn_run.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress.setValue(100)
        ok = sum(1 for r in results if r.get("ok"))
        self.stage_label.setText(f"完成：成功 {ok} / 共 {len(results)}，"
                                 f"总耗时 {total_sec:.1f} 秒")
        self._append_log("ok", f"全部结束，总耗时 {total_sec:.1f} 秒")
        self._append_log("info", f"输出目录：{self.out_edit.text()}")


def run_gui(initial_files: list[str] | None = None,
            autorun: bool = False) -> int:
    """启动桌面界面。

    initial_files / autorun 用于「把音频拖到 exe 图标上」这条入口：
    预填文件列表并自动开始处理，让用户能看到进度，而不是静默跑完。
    """
    from .config import load_config

    cfg = load_config()
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Song2MIDI")
    app.setStyleSheet(_QSS)
    win = MainWindow(cfg, initial_files=initial_files, autorun=autorun)
    win.show()
    return app.exec()
