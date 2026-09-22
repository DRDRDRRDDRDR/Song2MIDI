"""引擎层。

每个引擎只做一件事，统一约定：
    输入  (channels, samples) 的 float32 数组，或 wav 文件路径
    输出  音符事件列表（StemNotes），或分轨音频字典 {名称: (channels, samples)}

分离引擎（可互换，对外接口一致）：
    separate            Demucs htdemucs / htdemucs_ft（快，约 2 分钟/首）
    separate_roformer   BS-RoFormer 4stems（质量更好，CPU 极慢，需 GPU）

转录引擎：
    pitch_onnx   Basic Pitch ONNX 音高转录
    drums        onset 检测 + 频段分类 → 鼓组
    chords       chroma 模板匹配 → 和弦
"""

from __future__ import annotations

from typing import Any

from ._vendor import note_creation as bp_note_creation  # noqa: F401

__all__ = ["bp_note_creation", "create_separator", "SEPARATOR_ENGINES", "separator_choices"]

# 可用的分离引擎名
SEPARATOR_ENGINES = ("demucs", "roformer")

# 各引擎可选的权重档位，供界面展示
_DEMUCS_MODELS = {
    "htdemucs": "单模型四轨，最快（推荐日常使用）",
    "htdemucs_ft": "四模型精调，质量更好，约 4 倍耗时",
    "hdemucs_mmi": "早期版本，速度与质量居中",
    "mdx_extra": "MDX 架构，质量较高但占用大",
}
_ROFORMER_MODELS = {
    "bs_roformer": "BS-RoFormer 4 轨（Apache-2.0），质量最好，需 GPU 才实用",
}


def create_separator(cfg, engine: str | None = None, **kwargs: Any):
    """按配置创建分离器。

    两个实现对外接口完全一致（separate / model_name / device / sources / info），
    因此管线层不需要知道用的是哪一个。
    """
    eng = (engine or cfg.get("separator.engine", "demucs") or "demucs").lower()
    if eng == "roformer":
        from .separate_roformer import RoFormerSeparator

        return RoFormerSeparator(cfg, **kwargs)

    from .separate import DemucsSeparator

    return DemucsSeparator(cfg, **kwargs)


def separator_choices() -> dict[str, dict[str, str]]:
    """返回可选引擎与档位，供界面下拉框使用。"""
    return {"demucs": dict(_DEMUCS_MODELS), "roformer": dict(_ROFORMER_MODELS)}
