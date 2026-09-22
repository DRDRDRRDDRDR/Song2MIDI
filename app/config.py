"""配置加载。

原则：config.yaml 可以不存在或只写一部分，缺失项一律回落到代码内默认值，
保证程序在任何残缺配置下都能启动，不会因为一个键没写就崩掉。
"""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path
from typing import Any

__all__ = ["PROJECT_ROOT", "RESOURCE_ROOT", "DATA_ROOT", "load_config", "Config"]


def _detect_roots() -> tuple[Path, Path]:
    """返回 (资源根, 数据根)。

    打包成 exe 后必须把两者分开，否则会出问题：
      - 资源（模型、静态文件、config.yaml）随包分发，位于 PyInstaller 的
        解压根目录（sys._MEIPASS）。
      - 输出与中间产物必须写到 exe 同级的真实目录。若跟着资源走，
        onefile 模式下会被写进临时解压目录，程序退出即消失 ——
        用户会以为「跑完了但找不到文件」。
    """
    frozen = getattr(sys, "frozen", False)
    if frozen:
        resource = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
        data = Path(sys.executable).resolve().parent
        return resource, data
    root = Path(__file__).resolve().parent.parent
    return root, root


RESOURCE_ROOT, DATA_ROOT = _detect_roots()

# 向后兼容：此前多处引用 PROJECT_ROOT 表示项目根目录
PROJECT_ROOT = DATA_ROOT


def user_models_root() -> Path:
    """模型权重的系统固定存放位置。

    为什么模型不跟软件走：三个模型合计约 750 MB，而 CUDA 版 torch 本身已有
    2.5 GB —— 打在一起会让产物膨胀到 5 GB，超过 GitHub Release 的 2 GB 单文件
    上限，也没法用常规方式分发。外置之后：软件本体变小、模型可跨版本复用、
    用户按需下载。

    位置选 %LOCALAPPDATA%/Song2MIDI/models（Windows 上即 AppData\\Local\\Song2MIDI\\models）：
      - 符合 Windows 惯例（用户级数据放 LOCALAPPDATA，不污染安装目录）
      - 不需要管理员权限
      - 多个版本/多个 exe 副本可以共享同一份模型
    可用环境变量 SONG2MIDI_MODELS 覆盖（便于放到大容量盘）。
    """
    import os

    env = os.environ.get("SONG2MIDI_MODELS")
    if env:
        return Path(env).expanduser()
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
        return Path(base) / "Song2MIDI" / "models"
    # 非 Windows：遵循 XDG 惯例
    base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return Path(base) / "Song2MIDI" / "models"


MODELS_ROOT = user_models_root()

# 这些目录的内容随包分发、只读，优先从资源根解析
_BUNDLED_DIRS = {"models"}

DEFAULTS: dict[str, Any] = {
    "paths": {
        "models_dir": "models",
        "work_dir": "work",
        "out_dir": "out",
        "logs_dir": "logs",
    },
    "ffmpeg": {
        # 留空表示自动发现（会复用本机已有的 full build）
        "binary": "",
        "probe": "",
    },
    "audio": {
        # 中间产物统一采样率。Demucs 要求 44.1kHz，故以此为基准。
        "target_sr": 44100,
        "channels": 2,
        # 单次送入模型的片段长度（秒）。长曲分块可显著降低显存峰值。
        "segment_sec": 60,
        # 相邻片段的重叠，避免切点处音符被截断
        "segment_overlap_sec": 2,
    },
    "separator": {
        # 分离引擎：demucs（快）| roformer（质量更好，但必须 GPU 才实用）
        "engine": "demucs",
        # htdemucs: 单模型四轨，速度快；htdemucs_ft: 四模型精调，质量更好但慢约 4 倍
        "model": "htdemucs",
        "device": "auto",          # auto | cuda | cpu
        "shifts": 1,               # 随机移位平均次数，越大越稳越慢
        "overlap": 0.25,
        "jobs": 1,                 # Demucs 内部并行段数，受显存限制
        # 8GB 显存下同时跑几个分离任务。实测后调整。
        "max_concurrent": 1,

        # ---- BS-RoFormer ----
        # 官方推理流程含 4 倍重叠（num_overlap=4），CPU 实测 0.05x 实时，
        # 4 分钟的歌约需 79 分钟；GPU 上才具备实用性。
        "roformer_type": "bs_roformer",
        "roformer_repo": "SYH99999/bs_roformer_4stems_ft",
        "roformer_checkpoint": "bs_roformer_4stems_ft.pth",
        "roformer_config": "config.yaml",
        "roformer_checkpoint_bytes": 527245586,
    },
    "network": {
        # 本机 DNS 会把 huggingface.co 解析到 127.0.0.1，下载模型必须走本地代理。
        # 程序对下列端口逐个实测（连通性 + 速度），选最快的一条。
        # 环境变量里的 HTTPS_PROXY 也会作为候选参与。
        "proxy_ports": [9549, 33331, 7890, 7897, 10809, 10808, 1080, 2080,
                        8080, 20171, 8889],
    },
    "transcription": {
        "onset_threshold": 0.5,
        "frame_threshold": 0.3,
        "minimum_note_length_ms": 58.0,
        "minimum_frequency": None,   # None = 用模型默认下限
        "maximum_frequency": None,
        "melodia_trick": True,
        # 各分轨的转录策略
        "per_stem": {
            "vocals": {"mode": "polyphonic", "fmin": 65.0,  "fmax": 1500.0, "max_polyphony": 2},
            "bass":   {"mode": "monophonic",  "fmin": 28.0,  "fmax": 400.0,  "max_polyphony": 1},
            "other":  {"mode": "polyphonic",  "fmin": 55.0,  "fmax": 4000.0, "max_polyphony": 6},
            # 整首直转（跳过分离）时使用，钢琴曲走这条路径
            "mix":    {"mode": "polyphonic",  "fmin": 27.5,  "fmax": None,    "max_polyphony": 10},
        },
    },
    # 独奏检测：无鼓+无贝斯 → 跳过四轨分离（见 app/solo_detect.py）
    "solo_detect": {
        "high_ratio_limit": 0.02,   # 6k–16kHz 能量占比低于此值 → 无鼓/镲
        "low_ratio_limit": 0.03,    # 20–80Hz 能量占比低于此值 → 无贝斯
        "analysis_sr": 22050,
    },
    "drums": {
        "onset_sensitivity": 0.5,
        # GM 鼓组音高映射
        "mapping": {
            "kick": 36, "snare": 38, "closed_hat": 42,
            "open_hat": 46, "low_tom": 41, "high_tom": 48,
            "crash": 49, "ride": 51,
        },
        "kick_max_hz": 120.0,
        "snare_band_hz": [180.0, 900.0],
        "hat_min_hz": 5000.0,
    },
    "chords": {
        "enabled": True,
        "n_chroma": 12,
        "hop_sec": 0.1,
        # 和弦模板匹配的最小置信度，低于该值判定为无和弦
        "min_confidence": 0.35,
    },
    "postprocess": {
        "quantize": "1/16",        # off | 1/4 | 1/8 | 1/16 | 1/32
        "bpm": "auto",             # auto 或数字
        "min_note_ms": 50.0,
        "merge_gap_ms": 40.0,      # 间隔小于此值的同音高音符合并
        "merge_tolerance_semitones": 0,
        "remove_out_of_range": True,
        "velocity_from_energy": True,
    },
    "pool": {
        # 0 = 自动按 CPU 核心数推断
        "max_workers": 0,
        "gpu_workers": 1,
        "mp_start_method": "spawn",   # Windows 下 spawn 更稳
    },
    "server": {
        "host": "127.0.0.1",
        "port": 8756,
        "open_browser": True,
    },
    "output": {
        "midi_type": "multi_track",   # multi_track | per_stem | both
        "also_write_json": True,       # 附带音符 JSON，便于排查
        "program_map": {               # GM 音色
            "vocals": 53,   # Voice Oohs
            "bass": 33,     # Electric Bass (finger)
            "other": 1,     # Acoustic Grand Piano
            "drums": 0,     # 通道 10 鼓组
            "chords": 1,
        },
    },
    "logging": {
        "level": "INFO",
        "keep_intermediate": True,
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """递归合并：override 中的值覆盖 base，字典逐层合并而非整体替换。"""
    out = copy.deepcopy(base)
    for key, val in (override or {}).items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


class Config:
    """点号访问的配置对象：cfg.get("separator.model")。"""

    def __init__(self, data: dict[str, Any], source: Path | None = None) -> None:
        self._data = data
        self.source = source

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def section(self, name: str) -> dict[str, Any]:
        val = self._data.get(name)
        return val if isinstance(val, dict) else {}

    def path(self, dotted: str) -> Path:
        """取配置里的相对路径并解析为绝对路径。

        解析规则：
          - 绝对路径原样返回。
          - models 目录按「资源根 → 数据根 → 系统固定位置」依次探测，
            只要某处有内容就用它（见 user_models_root 的说明）。
          - out / work / logs 等可写目录一律落在数据根，保证用户能找到。
        """
        raw = self.get(dotted)
        if not raw:
            raise KeyError(f"配置项 {dotted} 为空")
        p = Path(raw)
        if p.is_absolute():
            return p

        name = p.parts[0] if p.parts else ""
        if name in _BUNDLED_DIRS:
            # 模型目录：按「随包 → 便携 → 系统固定位置」的顺序找。
            # 只要某个位置有内容就用它，从而兼容「老版本把模型放在 exe 同级」
            # 的用户，避免升级后又重下 750 MB。
            for cand in (RESOURCE_ROOT / p, DATA_ROOT / p):
                try:
                    if cand.is_dir() and any(cand.iterdir()):
                        return cand
                except OSError:
                    continue
            return MODELS_ROOT / Path(*p.parts[1:]) if len(p.parts) > 1 else MODELS_ROOT
        return DATA_ROOT / p

    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    def __repr__(self) -> str:
        return f"Config(source={self.source})"


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        import yaml  # type: ignore
    except ImportError:
        # yaml 缺失时尝试极简解析，保证程序仍可启动
        return _minimal_yaml(path)
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh.read())
    return data if isinstance(data, dict) else {}


def _minimal_yaml(path: Path) -> dict[str, Any]:
    """无 pyyaml 时的降级解析器：只支持两层缩进的 key: value。

    存在的意义是让「依赖尚未装完」这种中间状态也能跑起来，
    不因为少一个解析库就整条链路中断。
    """
    result: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, result)]
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        body = line.strip()
        if ":" not in body:
            continue
        key, _, val = body.partition(":")
        key, val = key.strip(), val.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if val == "":
            node: dict[str, Any] = {}
            parent[key] = node
            stack.append((indent, node))
        else:
            parent[key] = _coerce(val)
    return result


def _coerce(val: str) -> Any:
    low = val.lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    if low in ("null", "none", "~"):
        return None
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    return val.strip("'\"")


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    """加载配置。找不到文件时返回纯默认配置。

    查找顺序：显式路径 → 数据根（exe 同级，便于用户改）→ 资源根（随包分发）。
    把可写位置放在前面，用户在 exe 旁边放一份 config.yaml 就能覆盖默认值，
    而不必去动打包内部的资源。
    """
    if path:
        cfg_path = Path(path)
    else:
        candidates = [DATA_ROOT / "config.yaml", RESOURCE_ROOT / "config.yaml"]
        cfg_path = next((c for c in candidates if c.is_file()), candidates[0])

    user = _read_yaml(cfg_path)
    data = _deep_merge(DEFAULTS, user)
    cfg = Config(data, cfg_path if cfg_path.is_file() else None)

    for key in ("paths.models_dir", "paths.work_dir", "paths.out_dir", "paths.logs_dir"):
        try:
            cfg.path(key).mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
    return cfg
