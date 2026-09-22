"""音频摄取：任意格式 → 标准 wav，并做带缓存与体检。

这一层存在的意义是**把不确定性收拢在一处**。上游可能是 mp3 / flac /
m4a / 24bit wav / 带封面的容器，下游模型只接受固定采样率的 float 数组。
在中间做一次归一，后续所有引擎都不必再关心格式问题。

缓存策略：以「源文件路径 + 修改时间 + 大小 + 目标参数」的哈希为键。
歌曲文件通常不变，因此重复处理同一首歌时可直接命中，避免重复解码。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from .config import Config
from .ffmpeg_tools import FFmpegTools, get_tools

__all__ = ["AudioInfo", "Ingest", "SUPPORTED_EXTS"]

SUPPORTED_EXTS = {
    ".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".oga",
    ".opus", ".wma", ".aiff", ".aif", ".ape", ".alac", ".mp4", ".mkv", ".webm",
}

# 超过该时长给出提醒：分离与转录耗时随长度线性增长
_LONG_AUDIO_SEC = 900.0
# 低于该时长基本不可能是完整歌曲，多半是误选文件
_SHORT_AUDIO_SEC = 3.0


@dataclass
class AudioInfo:
    """解码产物的元信息。"""

    source_path: str
    wav_path: str
    duration_sec: float
    sample_rate: int
    channels: int
    source_codec: str | None = None
    source_container: str | None = None
    source_bits: int | None = None
    from_cache: bool = False
    warnings: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def duration_hms(self) -> str:
        total = int(round(self.duration_sec))
        return f"{total // 60}:{total % 60:02d}"


class IngestError(RuntimeError):
    """输入文件不可用。"""


class Ingest:
    def __init__(self, cfg: Config, tools: FFmpegTools | None = None) -> None:
        self.cfg = cfg
        self.tools = tools or get_tools(
            cfg.get("ffmpeg.binary") or None,
            cfg.get("ffmpeg.probe") or None,
        )
        self.work_dir = cfg.path("paths.work_dir")
        self.target_sr = int(cfg.get("audio.target_sr", 44100))
        self.channels = int(cfg.get("audio.channels", 2))

    # ---------- 校验 ----------

    def validate(self, src: str | Path) -> Path:
        p = Path(src)
        if not p.exists():
            raise IngestError(f"文件不存在: {p}")
        if not p.is_file():
            raise IngestError(f"不是文件: {p}")
        if p.suffix.lower() not in SUPPORTED_EXTS:
            raise IngestError(
                f"不支持的扩展名 {p.suffix}。支持: {', '.join(sorted(SUPPORTED_EXTS))}"
            )
        size = p.stat().st_size
        if size == 0:
            raise IngestError(f"文件为空: {p}")
        return p

    def inspect(self, src: str | Path) -> dict[str, Any]:
        """只做探测，不解码。用于 Web UI 上传后立即回显信息。"""
        p = self.validate(src)
        info = self.tools.probe(str(p))
        warnings = self._duration_warnings(info.get("duration_sec"))
        info["warnings"] = warnings
        info["file_size_mb"] = round(p.stat().st_size / 1048576, 2)
        info["hms"] = self._hms(info.get("duration_sec"))
        return info

    @staticmethod
    def _hms(sec: float | None) -> str:
        if not sec:
            return "--:--"
        s = int(round(sec))
        return f"{s // 60}:{s % 60:02d}"

    @staticmethod
    def _duration_warnings(duration: float | None) -> list[str]:
        out: list[str] = []
        if duration is None:
            out.append("无法读取时长，文件可能损坏")
            return out
        if duration > _LONG_AUDIO_SEC:
            out.append(
                f"音频长达 {duration / 60:.1f} 分钟，分离与转录耗时将显著增加，"
                "建议先用片段试跑"
            )
        if duration < _SHORT_AUDIO_SEC:
            out.append(f"音频仅 {duration:.1f} 秒，过短，可能不是完整歌曲")
        return out

    # ---------- 解码 ----------

    def _cache_key(self, p: Path) -> str:
        st = p.stat()
        raw = f"{p.resolve()}|{st.st_mtime_ns}|{st.st_size}|{self.target_sr}|{self.channels}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

    def prepare(self, src: str | Path, use_cache: bool = True,
                start: float | None = None, duration: float | None = None) -> AudioInfo:
        """把输入解码为项目内标准 wav。片段截取时不使用缓存。"""
        p = self.validate(src)
        meta = self.tools.probe(str(p))
        warnings = self._duration_warnings(meta.get("duration_sec"))

        is_segment = start is not None or duration is not None
        if is_segment:
            tag = f"seg_{int((start or 0) * 1000)}_{int((duration or 0) * 1000)}"
            wav = self.work_dir / "decoded" / f"{p.stem}_{tag}_{self.target_sr}.wav"
        else:
            wav = self.work_dir / "decoded" / f"{p.stem}_{self._cache_key(p)}_{self.target_sr}.wav"

        from_cache = False
        if use_cache and not is_segment and wav.is_file() and wav.stat().st_size > 1024:
            from_cache = True
        else:
            self.tools.decode_to_wav(
                str(p), str(wav),
                target_sr=self.target_sr,
                channels=self.channels,
                start=start, duration=duration,
            )

        # 以解码产物为准回读实际时长，比容器声明更可靠
        wav_meta = self.tools.probe(str(wav))
        real_dur = wav_meta.get("duration_sec") or meta.get("duration_sec") or 0.0

        if meta.get("n_audio_streams", 1) > 1:
            warnings.append(f"文件含 {meta['n_audio_streams']} 条音频流，已取第一条")

        return AudioInfo(
            source_path=str(p.resolve()),
            wav_path=str(wav),
            duration_sec=float(real_dur),
            sample_rate=self.target_sr,
            channels=self.channels,
            source_codec=meta.get("codec"),
            source_container=meta.get("container"),
            source_bits=meta.get("bits_per_sample"),
            from_cache=from_cache,
            warnings=warnings,
        )

    # ---------- 读取为数组 ----------

    def load_array(self, wav_path: str | Path, mono: bool = False,
                   sr: int | None = None):
        """读取 wav 为 float32 数组。

        返回 (data, sample_rate)。mono=False 时 data 形状为 (channels, samples)，
        与 Demucs / torchaudio 的约定一致，避免下游反复转置。
        """
        import numpy as np
        import soundfile as sf

        data, file_sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
        data = data.T  # (samples, channels) -> (channels, samples)
        if mono:
            data = data.mean(axis=0, keepdims=True)
        if sr is not None and sr != file_sr:
            data = resample(data, file_sr, sr)
            file_sr = sr
        return np.ascontiguousarray(data), file_sr

    def write_array(self, data, sr: int, dst: str | Path) -> str:
        """把 float32 数组写成 16bit wav。分轨产物落盘用这个。"""
        import numpy as np
        import soundfile as sf

        arr = np.asarray(data, dtype="float32")
        if arr.ndim == 2 and arr.shape[0] < arr.shape[1]:
            arr = arr.T  # (channels, samples) -> (samples, channels)
        out = Path(dst)
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out), np.clip(arr, -1.0, 1.0), sr, subtype="PCM_16")
        return str(out)

    # ---------- 任务清单 ----------

    def manifest(self, info: AudioInfo, extra: dict[str, Any] | None = None) -> Path:
        """把摄取结果写入 JSON 清单，便于事后复盘与断点续跑。"""
        payload = info.to_dict()
        if extra:
            payload.update(extra)
        dst = self.work_dir / "manifests" / f"{Path(info.source_path).stem}.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return dst


def resample(data, src_sr: int, dst_sr: int):
    """重采样。输入输出统一为 (channels, samples)。

    约定差异提醒：soxr 与 scipy 用 (samples, channels)，而 librosa 与
    torchaudio 用 (channels, samples)。本项目全程采用后者，因此在
    soxr 边界处必须转置，否则会把采样点数误判为声道数而报错。
    """
    import numpy as np

    arr = np.asarray(data, dtype=np.float32)
    squeeze = arr.ndim == 1
    if squeeze:
        arr = arr[None, :]

    def _finish(out):
        out = np.ascontiguousarray(out)
        return out[0] if squeeze else out

    if src_sr == dst_sr or arr.shape[-1] == 0:
        return _finish(arr)

    # 1) soxr：质量与速度最佳，需要转置成 (samples, channels)
    try:
        import soxr  # type: ignore

        return _finish(soxr.resample(arr.T, src_sr, dst_sr, quality="HQ").T)
    except Exception:
        pass

    # 2) librosa：本身即 (channels, samples) 约定
    try:
        import librosa  # type: ignore

        return _finish(librosa.resample(arr, orig_sr=src_sr, target_sr=dst_sr))
    except Exception:
        pass

    # 3) 线性插值兜底。质量一般，但保证在依赖残缺时链路不中断。
    n_out = int(round(arr.shape[-1] * dst_sr / src_sr))
    if n_out <= 0:
        return _finish(np.zeros((arr.shape[0], 1), dtype=np.float32))
    idx = np.linspace(0, arr.shape[-1] - 1, n_out)
    left = np.floor(idx).astype(np.int64)
    right = np.minimum(left + 1, arr.shape[-1] - 1)
    frac = (idx - left).astype(np.float32)
    return _finish(arr[..., left] * (1 - frac) + arr[..., right] * frac)
