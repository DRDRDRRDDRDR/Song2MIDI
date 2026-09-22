# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 打包配置（onedir 模式）。

路径基准说明（踩过的坑）：
    PyInstaller 解析 spec 里的相对路径时，以 **spec 文件所在目录**为基准，
    而不是当前工作目录。本 spec 位于 build/ 下，若直接写 "main.py"，
    它会去找 build/main.py 并报 "script not found"。
    因此下面所有路径统一用 ROOT（项目根）拼绝对路径。

为什么只用 onedir：
    本包含 CUDA 版 torch（约 3.9 GB 的 DLL）+ BS-RoFormer 权重（503 MB），
    整体约 5.2 GB。onefile 每次启动都要把全部内容解压到临时目录，
    启动会长达数分钟、磁盘占用翻倍，实际不可用。

排错提示：若打包后启动报 ModuleNotFoundError，把缺失模块名加进
hiddenimports 即可。PyInstaller 无法静态分析出动态导入与插件式加载。
"""

from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules, collect_data_files

# spec 所在目录是 build/，项目根在它上一层
ROOT = Path(SPECPATH).resolve().parent

# ---------------------------------------------------------------- 资源文件
datas = []


def add_data(rel: str, dest: str | None = None) -> None:
    """把项目内的相对路径加入打包数据。存在才加，避免打包中止。

    dest 的语义要注意：PyInstaller 会把 **源目录的内容** 放进 dest 目录。
    所以 dest 必须写到与源同级的完整相对路径，否则会少一层。
    曾把 ("app/static", "app") 写成 dest="app"，结果 index.html 落在了
    _internal/app/ 而不是 _internal/app/static/。
    """
    src = ROOT / rel
    if src.exists():
        datas.append((str(src), dest if dest is not None else rel.replace("\\", "/")))


add_data("config.yaml", ".")
add_data("app/static", "app/static")

# ---- 模型权重的打包策略：只打最小的那个，其余外置 ----
#
# 为什么不全部随包分发：三个模型合计约 750 MB，而 CUDA 版 torch 本身已有 2.5 GB，
# 打在一起会让产物膨胀到 5 GB —— 超过 GitHub Release 的 2 GB 单文件上限，
# 也没法用常规方式分发。改为由软件内的「模型浏览器」按需下载到
# %LOCALAPPDATA%\Song2MIDI\models（见 app/models_registry.py）。
#
# 为什么仍然打进 nmp.onnx：它只有 225 KB，却是转录功能的**必需项** ——
# 让用户为了 225 KB 先去点一次下载不合理；而且它没有独立直链
# （取自 basic-pitch wheel），放在包内最省事。其余模型全部外置。
add_data("assets/nmp.onnx", "models")

# ---- 依赖包自带的非 .py 数据文件 ----
#
# 必须显式收集，否则会缺文件且报错信息完全不提「是打包漏了」。
# 实测踩到：demucs 的 remote/files.txt 缺失 —— 该文件记录预训练权重的
# 下载地址清单，get_model() 一读就抛
#   FileNotFoundError: ...\_internal\demucs\remote\files.txt
# 而上层提示却是「需要下载约 80 MB 权重」，把排查方向带偏到网络上。
for _pkg in ("demucs", "msst", "librosa", "pretty_midi", "soundfile",
             "resampy", "soxr", "einops", "omegaconf", "ml_collections"):
    try:
        datas += collect_data_files(_pkg)
    except Exception:
        pass

# ---------------------------------------------------------------- 隐式导入
hiddenimports = []

# 这些包依赖运行时动态导入，静态分析发现不了
for _pkg in ("demucs", "msst", "einops", "einx", "rotary_embedding_torch",
             "hyper_connections", "pope_pytorch", "torch_einops_utils",
             "beartype", "omegaconf", "ml_collections", "antlr4", "absl",
             "julius", "lameenc", "openunmix", "treetable",
             "retrying", "submitit", "cloudpickle", "soxr", "resampy",
             "numba", "llvmlite", "sklearn", "librosa", "soundfile",
             "pretty_midi", "onnxruntime", "torchaudio", "einops"):
    try:
        hiddenimports += collect_submodules(_pkg)
    except Exception:
        pass

# 本项目自身的子模块（tools 在冻结环境里要以模块方式导入）
hiddenimports += [
    "tools", "tools.check_env", "tools.run_cli", "tools.transcribe_probe",
    "tools.test_drums", "tools.fetch_wheels", "tools.compare_separators",
    "tools.roformer_gpu_bench",
    "app", "app.config", "app.gui", "app.pipeline", "app.pool", "app.ingest",
    "app.notes", "app.export", "app.postprocess", "app.ffmpeg_tools",
    "app.net", "app.api", "app.solo_detect", "app.preprocess",
    "app.engines.pitch_piano",
    "app.engines", "app.engines.separate", "app.engines.separate_roformer",
    "app.engines.pitch_onnx", "app.engines.drums", "app.engines.chords",
    "app.engines._vendor", "app.engines._vendor.constants",
    "app.engines._vendor.note_creation",
]

# 钢琴转录引擎所需的第三方包。
# 这些是纯 Python 包，但入口分散，显式列出更稳妥。
hiddenimports += [
    "piano_transcription_inference",
    "piano_transcription_inference.inference",
    "piano_transcription_inference.models",
    "piano_transcription_inference.utilities",
    "piano_transcription_inference.config",
    "piano_transcription_inference.pytorch_utils",
    "torchlibrosa",
    "torchlibrosa.stft",
    "audioread",
]

# torch 体积巨大，只收必要子模块，不做 collect_submodules（会拖入测试与文档）
hiddenimports += [
    "torch", "torch._C", "torch.nn", "torch.nn.functional",
    "torch.utils.data", "torch.utils.checkpoint", "torch.jit",
    "torch.backends", "torch.backends.cuda", "torch.backends.cudnn",
    "torch.cuda", "torch.cuda.amp", "torch.fft", "torch.linalg",
    "torch.hub", "torch.serialization", "torch.storage",
]

# ---------------------------------------------------------------- 排除
#
# 重要教训：不要排除 torch 的任何内部子包来省体积。
# 曾把 torch._inductor / torch._dynamo / torch.distributed 加进 excludes，
# 以为「纯推理用不到编译栈」，结果 `import torch` 直接失败 ——
# 这三者是被 torch 顶层导入的（PyInstaller 的警告文件里标注为 top-level），
# 排除后整个 GPU 链路失效，而体积只省下约 44 MB（相对于 4.8 GB 毫无意义）。
excludes = [
    "tkinter",              # 本环境没有，且与 PySide6 无关
    "matplotlib.backends.backend_qt5agg",
    "matplotlib.backends.backend_qtagg",
    "IPython", "jupyter", "notebook", "pytest", "sphinx", "dask",
    "PyQt5", "PyQt6", "PySide2",       # 只用了 PySide6
    "torchvision", "torchtext",        # 本项目不用
    "torch.utils.tensorboard",         # 依赖 tensorboard，且本项目不用
    "tensorboard", "triton",           # 训练/编译用，Windows 上也没有
]


# ---------------------------------------------------------------- 构建
a = Analysis(
    [str(ROOT / "main.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    # 关键：在导入 torch 之前把 torch/lib 注册为 DLL 搜索目录，
    # 否则冻结后 import torch 会抛 WinError 126 且不提示缺哪个 DLL
    runtime_hooks=[str(ROOT / "build" / "rthook_torch_dll.py")],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Song2MIDI",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,          # UPX 对 torch 的 CUDA DLL 有实际风险，关闭
    console=False,      # GUI 程序，不弹黑窗
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="Song2MIDI",
)
