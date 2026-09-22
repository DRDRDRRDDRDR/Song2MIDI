"""转录前的音频预处理。

**关于降噪的取舍（实测数据说话）**：

本项目实测过两类降噪，结论是「不是所有降噪都值得加」：

    方案            信号改动   碎片率        音高保留
    noisereduce     58.1%     1.3%→1.5%↑    丢失 E1、A5（音域 28–88 缩到 38–88）
    高通 50Hz        3.9%     1.3% 不变      100%

noisereduce 改动了 58% 的信号却毫无收益，还削掉了钢琴低音区 —— 因为这类
谱门控降噪多为**语音**设计，会把乐器赖以识别音高的高频泛音一起减掉。

因此这里**只提供高通**，它是唯一经实测确认无害的处理：
低频嗡声（房间嗡鸣、电源哼声、空调）集中在几十 Hz，而钢琴最低音 A0 是 27.5 Hz、
常规演奏很少用到 50Hz 以下，滤掉几乎不伤内容。
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["apply_highpass", "PREPROCESS_VERSION"]

PREPROCESS_VERSION = 1


def apply_highpass(audio: np.ndarray, sr: int, cutoff_hz: float,
                   order: int = 4) -> tuple[np.ndarray, dict[str, Any]]:
    """对音频施加 Butterworth 高通滤波。

    audio 支持 (channels, samples) 或 (samples,)。
    返回 (处理后音频, 说明信息)。cutoff_hz <= 0 时原样返回。
    """
    y = np.asarray(audio, dtype=np.float32)
    info: dict[str, Any] = {"highpass_hz": float(cutoff_hz), "applied": False}

    if not cutoff_hz or cutoff_hz <= 0:
        return y, info
    if sr <= 0:
        return y, info

    # 截止频率必须低于奈奎斯特，否则滤波器设计失败
    nyq = sr / 2.0
    if cutoff_hz >= nyq * 0.95:
        info["skipped"] = f"截止频率 {cutoff_hz}Hz 接近奈奎斯特 {nyq:.0f}Hz，已跳过"
        return y, info

    try:
        from scipy.signal import butter, sosfilt

        sos = butter(order, float(cutoff_hz), btype="highpass", fs=sr, output="sos")
        out = sosfilt(sos, y, axis=-1).astype(np.float32)
    except Exception as e:
        info["error"] = f"{e.__class__.__name__}: {e}"
        return y, info

    # 改动量：用于在日志里如实报告「这一步到底改了多少信号」
    n = min(y.shape[-1], out.shape[-1])
    diff = float(np.sqrt(((out[..., :n] - y[..., :n]) ** 2).mean()))
    ref = float(np.sqrt((y[..., :n] ** 2).mean())) or 1e-12
    info.update({
        "applied": True,
        "order": order,
        "changed_rms": round(diff, 6),
        "changed_pct": round(diff / ref * 100, 2),
    })
    return out, info
