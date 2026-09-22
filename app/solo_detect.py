"""独奏 / 无节奏组曲目的自动检测。

背景：Demucs 的四轨分离（vocals/drums/bass/other）是为流行乐编制训练的。
纯钢琴曲、吉他独奏、弦乐独奏这类没有鼓和贝斯的曲目被强行分离时，
会把琴槌瞬态误判成鼓、把旋律剥散 —— 实测一首无鼓钢琴曲被分离出
490 个假鼓音符。因此有必要在处理前自动判断「是否该跳过分离」。

判定依据（本机实测的频段能量分布）：
    钢琴独奏：6k–16kHz 占比 0.33%（无镲/鼓）、20–80Hz 占比 1.39%（无贝斯）
    流行乐  ：这两个频段都明显更高。

阈值刻意设得保守 —— 宁可把「接近独奏」判成需要分离，也不要漏判，
因为误跳过分离的代价（丢失编配信息）大于误做一次分离（多花时间但结果仍对）。
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["detect_solo", "SOLO_DETECT_VERSION"]

SOLO_DETECT_VERSION = 1


def detect_solo(audio: np.ndarray, sr: int,
                high_ratio_limit: float = 0.02,
                low_ratio_limit: float = 0.03,
                analysis_sr: int = 22050) -> dict[str, Any]:
    """判断音频是否「无鼓且无贝斯」，建议跳过四轨分离。

    返回字典：
        solo       bool    是否判为独奏
        confidence float   0~1，越高越确信（基于频段占比偏离阈值的程度）
        metrics    各频段能量占比与阈值，供界面展示
        reason     人类可读的判定说明
    """
    import librosa

    y = np.asarray(audio, dtype=np.float32)
    if y.ndim == 2:
        y = y.mean(axis=0)              # 合成单声道，检测只关心整体频谱
    y = np.ascontiguousarray(y.reshape(-1))

    if sr != analysis_sr:
        try:
            y = librosa.resample(y, orig_sr=sr, target_sr=analysis_sr)
        except Exception:
            analysis_sr = sr

    # 频段能量分布
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=analysis_sr, n_fft=2048)
    total = S.sum() + 1e-9

    def band(lo: float, hi: float) -> float:
        m = (freqs >= lo) & (freqs < hi)
        return float(S[m].sum() / total)

    low = band(20, 80)       # 贝斯/底鼓
    high = band(6000, 16000)  # 镲片/高频泛音

    # 打击成分占比（HPSS），作为交叉验证
    try:
        y_h, y_p = librosa.effects.hpss(y)
        percussive = float(np.sqrt((y_p ** 2).mean()) /
                           (np.sqrt((y_h ** 2).mean()) + np.sqrt((y_p ** 2).mean()) + 1e-9))
    except Exception:
        percussive = 0.0

    no_drums = high < high_ratio_limit
    no_bass = low < low_ratio_limit
    solo = no_drums and no_bass

    # 置信度：离阈值越远越确信
    if solo:
        c_high = (high_ratio_limit - high) / high_ratio_limit
        c_low = (low_ratio_limit - low) / low_ratio_limit
        confidence = float(np.clip(min(c_high, c_low), 0.0, 1.0))
    else:
        confidence = 0.0

    if solo:
        reason = f"高频占比 {high:.1%}（无鼓/镲）、次低频占比 {low:.1%}（无贝斯）"
    elif no_drums:
        reason = f"无鼓/镲，但次低频占比 {low:.1%} 偏高（疑似有贝斯）"
    elif no_bass:
        reason = f"无贝斯，但高频占比 {high:.1%} 偏高（疑似有鼓/镲）"
    else:
        reason = f"含鼓（高频 {high:.1%}）与贝斯（次低频 {low:.1%}），需分离"

    return {
        "version": SOLO_DETECT_VERSION,
        "solo": solo,
        "confidence": confidence,
        "no_drums": bool(no_drums),
        "no_bass": bool(no_bass),
        "percussive_ratio": round(percussive, 4),
        "high_ratio": round(high, 4),
        "low_ratio": round(low, 4),
        "high_ratio_limit": high_ratio_limit,
        "low_ratio_limit": low_ratio_limit,
        "analysis_sr": analysis_sr,
        "reason": reason,
    }
