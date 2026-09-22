"""环境自检工具。

用途：一行命令确认本机是否具备运行条件，出问题时先跑它，把排查范围收敛到具体环节。
    python tools/check_env.py
    python tools/check_env.py --json     # 输出机器可读结果

检查项按「不通过就必须修」的严重程度分三级：
    FAIL  阻塞性缺失，程序无法运行
    WARN  可降级运行，但功能受限
    OK    正常
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 打包后必须区分：模型等只读资源在资源根，输出目录在数据根（exe 同级）
try:
    from app.config import DATA_ROOT, RESOURCE_ROOT
except Exception:  # 极端情况下退回源码布局
    DATA_ROOT = RESOURCE_ROOT = ROOT

results: list[dict[str, str]] = []


def add(level: str, item: str, detail: str) -> None:
    results.append({"level": level, "item": item, "detail": detail})


def check_python() -> None:
    v = sys.version_info
    txt = f"{v.major}.{v.minor}.{v.micro}"
    if v >= (3, 11):
        add("OK", "Python 版本", f"{txt}（{sys.executable}）")
    else:
        add("FAIL", "Python 版本", f"{txt} 过低，需要 3.11+")


def check_modules() -> None:
    # 必需：缺失即无法工作
    required = {
        "numpy": "数值计算基础",
        "scipy": "信号处理",
        "soundfile": "音频读写",
        "librosa": "节拍/和弦/重采样",
        "pretty_midi": "MIDI 写出",
        "onnxruntime": "音高转录音推理",
    }
    # 可选：缺失只影响部分功能
    optional = {
        "torch": "Demucs 音源分离（四轨方案必需）",
        "torchaudio": "Demucs 音频前端",
        "demucs": "四轨分离模型",
        "fastapi": "Web UI",
        "uvicorn": "Web UI 服务",
        "yaml": "YAML 配置解析（缺失时降级为内置解析器）",
        "mido": "MIDI 辅助读写",
        "soxr": "高质量重采样",
        "sklearn": "scikit-learn，聚类/分类辅助",
    }

    for name, why in required.items():
        try:
            mod = importlib.import_module(name)
            ver = getattr(mod, "__version__", "?")
            add("OK", f"模块 {name}", f"v{ver} — {why}")
        except Exception as e:
            add("FAIL", f"模块 {name}", f"导入失败: {e.__class__.__name__}: {e} — {why}")

    for name, why in optional.items():
        try:
            mod = importlib.import_module(name)
            ver = getattr(mod, "__version__", "?")
            add("OK", f"模块 {name}", f"v{ver} — {why}")
        except ImportError as e:
            # 只有 ModuleNotFoundError 才是真的没装
            if isinstance(e, ModuleNotFoundError) and e.name == name:
                add("WARN", f"模块 {name}", f"未安装 — {why}")
            else:
                add("FAIL", f"模块 {name}",
                    f"导入失败（该模块本身存在，是它依赖的东西缺）: "
                    f"{e.__class__.__name__}: {e}")
        except Exception as e:
            # 非 ImportError 的异常尤其重要：例如 Windows 上 DLL 找不到
            # 会抛 WinError 126，报成「未安装」会把排查方向带偏
            add("FAIL", f"模块 {name}",
                f"导入时报错: {e.__class__.__name__}: {e}")


def check_ffmpeg() -> None:
    try:
        from app.ffmpeg_tools import get_tools

        tools = get_tools(refresh=True)
    except Exception as e:
        add("FAIL", "ffmpeg", f"未找到可用 ffmpeg: {e}")
        return

    add("OK", "ffmpeg", tools.ffmpeg)
    add("OK", "ffprobe", tools.ffprobe)
    add("OK", "ffmpeg 版本", tools.version)

    try:
        caps = tools.capabilities()
        missing = [k for k, v in caps.items() if not v]
        if missing:
            add("WARN", "ffmpeg 编解码能力", f"缺少: {', '.join(missing)}")
        else:
            add("OK", "ffmpeg 编解码能力", f"{len(caps)} 项全部齐备")
    except Exception as e:
        add("WARN", "ffmpeg 编解码能力", f"自检失败: {e}")


def check_model() -> None:
    model = RESOURCE_ROOT / "models" / "nmp.onnx"
    if not model.is_file():
        model = DATA_ROOT / "models" / "nmp.onnx"
    if not model.is_file():
        add("FAIL", "Basic Pitch ONNX 模型", f"缺失: {model}")
        return
    size = model.stat().st_size
    add("OK", "Basic Pitch ONNX 模型", f"{model.name} ({size:,} 字节)")

    try:
        import numpy as np
        import onnxruntime as ort

        sess = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0]
        add("OK", "ONNX 输入", f"{inp.name} shape={inp.shape} type={inp.type}")
        for o in sess.get_outputs():
            add("OK", "ONNX 输出", f"{o.name} shape={o.shape}")

        # 用零输入实跑一次，确认推理链路真的通，而不只是文件存在
        dummy = np.zeros((1, 43844, 1), dtype=np.float32)
        outs = sess.run(None, {inp.name: dummy})
        add("OK", "ONNX 推理自检", f"实跑通过，输出 {len(outs)} 个张量，"
                                  f"首个 shape={tuple(outs[0].shape)}")
    except Exception as e:
        add("FAIL", "ONNX 推理自检", f"{e.__class__.__name__}: {e}")


def check_gpu() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            add("OK", "CUDA", f"{name}，显存 {total:.1f} GB，torch {torch.__version__}")
        else:
            add("WARN", "CUDA", f"torch {torch.__version__} 未检测到可用 GPU，将使用 CPU")
    except ImportError:
        add("WARN", "CUDA", "torch 未安装，无法检测 GPU（四轨分离需要 torch）")
    except Exception as e:
        add("WARN", "CUDA", f"检测失败: {e}")


def check_dirs() -> None:
    """检查可写目录。不存在就顺手创建 —— 它们本来就是程序自动建的，
    只报「不存在」会让用户以为是故障。"""
    for name in ["models", "work", "out", "logs"]:
        p = DATA_ROOT / name
        created = ""
        if not p.is_dir():
            try:
                p.mkdir(parents=True, exist_ok=True)
                created = "（已自动创建）"
            except Exception as e:
                add("FAIL", f"目录 {name}/", f"无法创建 {p}：{e}")
                continue
        add("OK", f"目录 {name}/", f"{p}{created}")


def check_demucs() -> None:
    """检查 Demucs 的权重文件与包内数据文件。

    这个检查是补上的 —— 曾出现打包后 `demucs/remote/files.txt` 缺失
    （该文件记录预训练权重的下载地址清单，是包内数据文件而非 .py，
     PyInstaller 不会自动收集），而报错信息说的是「需要下载约 80 MB 权重」，
    把排查方向误导到网络上。有这道检查就能直接指出真正缺的是什么。
    """
    import importlib

    try:
        demucs = importlib.import_module("demucs")
        pkg_dir = Path(demucs.__file__).resolve().parent
    except Exception as e:
        add("WARN", "Demucs 包", f"未安装或无法导入：{e}")
        return

    remote = pkg_dir / "remote"
    files_txt = remote / "files.txt"
    if not files_txt.is_file():
        add("FAIL", "Demucs 包内数据",
            f"缺少 {files_txt} —— 这是包内数据文件，打包时必须用 "
            f"collect_data_files('demucs') 收集，否则模型加载会失败")
        return
    yamls = sorted(remote.glob("*.yaml"))
    add("OK", "Demucs 包内数据",
        f"files.txt + {len(yamls)} 个模型定义 yaml @ {remote}")

    # 权重缓存：torch.hub 会放到 $TORCH_HOME/hub/checkpoints/
    import os

    torch_home = os.environ.get("TORCH_HOME")
    cands = []
    if torch_home:
        cands.append(Path(torch_home) / "hub" / "checkpoints")
    cands += [RESOURCE_ROOT / "models" / "torch_hub" / "hub" / "checkpoints",
              DATA_ROOT / "models" / "torch_hub" / "hub" / "checkpoints"]

    for d in cands:
        if d.is_dir():
            th = sorted(d.glob("*.th")) + sorted(d.glob("*.pt"))
            if th:
                total = sum(f.stat().st_size for f in th) / 1048576
                add("OK", "Demucs 权重", f"{len(th)} 个文件，共 {total:.1f} MB @ {d}")
                return
    add("WARN", "Demucs 权重",
        "未找到缓存的权重文件，首次使用时会从 dl.fbaipublicfiles.com 下载（约 80 MB/档）")


def check_piano_engine() -> None:
    """检查钢琴专用转录引擎（ByteDance 高分辨率钢琴转录）。

    这条例行检查有必要：该 pip 包的 setup.py **漏声明了 audioread 依赖**，
    缺它时 import 会在中间某处炸，报错与真实原因无关；
    另外它用 wget 下载模型，Windows 上没有 wget 会静默失败，
    表现为「推理时找不到权重文件」。
    """
    import importlib

    try:
        importlib.import_module("piano_transcription_inference")
    except ImportError:
        add("WARN", "钢琴引擎包", "未安装 —— 将只用通用引擎 Basic Pitch。"
                                "安装：pip install piano-transcription-inference audioread")
        return
    except Exception as e:
        add("FAIL", "钢琴引擎包", f"存在但导入失败：{e.__class__.__name__}: {e}")
        return

    # 未声明的依赖，单独确认
    missing = []
    for m in ("torchlibrosa", "audioread"):
        try:
            importlib.import_module(m)
        except Exception:
            missing.append(m)
    if missing:
        add("FAIL", "钢琴引擎依赖",
            f"缺少 {'、'.join(missing)} —— 其中 audioread 是该包 setup.py 漏声明的。"
            f"安装：pip install {' '.join(missing)}")
        return
    add("OK", "钢琴引擎依赖", "piano_transcription_inference + torchlibrosa + audioread")

    # 权重
    from app.config import load_config
    from app.engines.pitch_piano import PianoTranscriber

    try:
        tr = PianoTranscriber(load_config())
        ck = Path(tr.checkpoint_path())
    except Exception as e:
        add("WARN", "钢琴引擎权重", f"无法确定权重路径：{e.__class__.__name__}: {e}")
        return
    if ck.is_file() and ck.stat().st_size > 1.6e8:
        add("OK", "钢琴引擎权重",
            f"{ck.name}（{ck.stat().st_size/1048576:.0f} MB）@ {ck.parent}")
    elif ck.is_file():
        add("FAIL", "钢琴引擎权重", f"文件不完整（{ck.stat().st_size/1048576:.1f} MB，"
                                    f"应为约 164 MB）：{ck}")
    else:
        add("WARN", "钢琴引擎权重",
            f"未找到：{ck} —— 该库用 wget 下载模型，Windows 上会失败，"
            f"需手动放置权重文件")


def check_roformer() -> None:
    """检查 BS-RoFormer 权重是否就位（可选的增强引擎）。"""
    import importlib

    try:
        msst = importlib.import_module("msst.inference")
        add("OK", "模块 msst", "已安装（BS-RoFormer 分离可用）")
    except Exception:
        add("WARN", "模块 msst", "未安装 — BS-RoFormer 分离不可用（Demucs 不受影响）")
        return

    cands = [
        RESOURCE_ROOT / "models" / "roformer" / "bs_roformer_4stems_ft",
        DATA_ROOT / "models" / "roformer" / "bs_roformer_4stems_ft",
    ]
    for d in cands:
        ck = d / "bs_roformer_4stems_ft.pth"
        cf = d / "config.yaml"
        if ck.is_file() and cf.is_file():
            add("OK", "BS-RoFormer 权重",
                f"{ck.stat().st_size/1048576:.1f} MB @ {d}")
            return
    add("WARN", "BS-RoFormer 权重",
        "未找到，首次使用时会自动从 HuggingFace 下载（约 503 MB）")


LEVEL_ORDER = {"FAIL": 0, "WARN": 1, "OK": 2}
LEVEL_MARK = {"FAIL": "[FAIL]", "WARN": "[WARN]", "OK": "[ OK ]"}


def main() -> int:
    ap = argparse.ArgumentParser(description="Song2MIDI 环境自检")
    ap.add_argument("--json", action="store_true", help="以 JSON 输出")
    args = ap.parse_args()

    # 必须清空：results 是模块级列表，同一进程内多次调用 main() 会不断累加。
    # 界面启动时会自动自检一次，用户再点「环境自检」按钮就是第二次 ——
    # 不清空的话结果会翻倍，计数与分组都会失真。
    results.clear()

    check_python()
    check_dirs()
    check_ffmpeg()
    check_modules()
    check_model()
    check_demucs()
    check_roformer()
    check_piano_engine()
    check_gpu()

    fails = [r for r in results if r["level"] == "FAIL"]
    warns = [r for r in results if r["level"] == "WARN"]

    if args.json:
        payload = json.dumps({"results": results, "fail": len(fails),
                              "warn": len(warns)},
                             ensure_ascii=False, indent=2)
        # JSON 是给机器读的，编码必须确定 —— 不能跟随系统默认编码
        # （中文 Windows 上是 GBK）。否则跨进程读取时一侧 GBK、一侧 UTF-8，
        # 中文就会变成「�汾」这类乱码。这里直接往 buffer 写 UTF-8 字节，
        # 绕开文本层的编码设置。
        try:
            sys.stdout.buffer.write(payload.encode("utf-8"))
            sys.stdout.buffer.flush()
        except (AttributeError, OSError):
            print(payload)
        return 1 if fails else 0

    print("=" * 68)
    print("Song2MIDI 环境自检")
    print("=" * 68)
    for level in ("FAIL", "WARN", "OK"):
        group = [r for r in results if r["level"] == level]
        if not group:
            continue
        for r in group:
            print(f"{LEVEL_MARK[level]} {r['item']}")
            print(f"        {r['detail']}")
        print("-" * 68)

    print(f"结论：{len(fails)} 项阻塞，{len(warns)} 项警告，"
          f"{len(results) - len(fails) - len(warns)} 项正常")
    if fails:
        print("\n必须先解决以下问题才能运行：")
        for r in fails:
            print(f"  - {r['item']}: {r['detail']}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
