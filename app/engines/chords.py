"""和弦识别：chroma 特征 + 模板匹配。

输出的是「和弦音」而非「和弦标签」—— 因为目标是 MIDI，需要的是具体音高。
标签（如 Cmaj7）一并保留在元信息里，便于人工核对。

实现要点：
    1. chroma 用 CQT 计算，对低频有更好的频率分辨率，和弦根音判定更稳。
    2. 逐帧匹配 12 个根音 × 若干和弦类型，取相关度最高者。
    3. 时间平滑：中值滤波 + 最短持续时间约束，消除逐帧抖动导致的
       「每个和弦只出现一帧」这种在 MIDI 里毫无意义的碎片。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..config import Config
from ..notes import NoteEvent, StemNotes

__all__ = ["ChordRecognizer", "ChordSegment", "CHORD_TEMPLATES"]

_PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# 和弦模板：音程度数 + 权重（根音权重略高，引导匹配偏向根音明确的解释）
CHORD_TEMPLATES: dict[str, tuple[tuple[int, ...], float]] = {
    "maj":      ((0, 4, 7), 1.00),
    "min":      ((0, 3, 7), 1.00),
    "dim":      ((0, 3, 6), 0.85),
    "aug":      ((0, 4, 8), 0.80),
    "sus4":     ((0, 5, 7), 0.85),
    "sus2":     ((0, 2, 7), 0.80),
    "maj7":     ((0, 4, 7, 11), 0.90),
    "min7":     ((0, 3, 7, 10), 0.90),
    "dom7":     ((0, 4, 7, 10), 0.90),
    "min6":     ((0, 3, 7, 9), 0.75),
    "maj6":     ((0, 4, 7, 9), 0.75),
    "halfdim7": ((0, 3, 6, 10), 0.75),
}

# 和弦音在 MIDI 中的落位音区。落在钢琴中音区，便于在 DAW 里直接看与听。
CHORD_BASE_OCTAVE = 4      # 根音落在 C4 附近
CHORD_NOTE_DURATION = 0.5  # 单帧和弦的默认时值（秒）


@dataclass
class ChordSegment:
    start: float
    end: float
    label: str
    root: int          # 0-11，相对 C 的半音数
    quality: str
    confidence: float
    pitches: list[int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": round(self.start, 3), "end": round(self.end, 3),
            "label": self.label, "root": self.root, "quality": self.quality,
            "confidence": round(self.confidence, 3), "pitches": self.pitches,
        }


def _build_templates() -> tuple[np.ndarray, list[tuple[int, str]]]:
    """预计算所有 (根音 × 和弦类型) 的归一化模板向量。"""
    vectors: list[np.ndarray] = []
    labels: list[tuple[int, str]] = []
    for root in range(12):
        for quality, (degrees, _weight) in CHORD_TEMPLATES.items():
            v = np.zeros(12, dtype=np.float32)
            for d in degrees:
                v[(root + d) % 12] = 1.0
            v /= np.linalg.norm(v)
            vectors.append(v)
            labels.append((root, quality))
    return np.stack(vectors), labels


class ChordRecognizer:
    def __init__(self, cfg: Config) -> None:
        sec = cfg.section("chords")
        self.enabled = bool(sec.get("enabled", True))
        self.hop_sec = float(sec.get("hop_sec", 0.1))
        self.min_confidence = float(sec.get("min_confidence", 0.35))
        # 和弦最短持续时间。实测钢琴曲中和弦切换间隔中位约 549ms，
        # 若最短只要求 250ms，会把大量短段留下，听感上「和弦一直在变」。
        # 调大到 400~600ms 可明显降密；默认仍为 250 以不改变既有行为。
        self.min_duration_ms = float(sec.get("min_duration_ms", 250.0))
        # 短段归并：A-B-A 中间那个很短的 B，多半是琶音/装饰音导致的误判
        self.merge_short_ms = float(sec.get("merge_short_ms", 0.0))
        self.sr = int(cfg.get("audio.target_sr", 44100))
        self.hop_length = max(1, int(round(self.hop_sec * self.sr)))
        self._templates, self._labels = _build_templates()

    # ---------- chroma ----------

    def _chroma(self, mono: np.ndarray) -> np.ndarray:
        """计算 chroma。优先 CQT（低频分辨率好），失败则退回 STFT 映射。"""
        try:
            import librosa

            c = librosa.feature.chroma_cqt(
                y=mono.astype(np.float32), sr=self.sr, hop_length=self.hop_length,
            )
            return np.asarray(c, dtype=np.float32)
        except Exception:
            return self._chroma_from_stft(mono)

    def _chroma_from_stft(self, mono: np.ndarray) -> np.ndarray:
        """STFT 幅度谱按频率映射到 12 个音级。不依赖任何第三方实现。"""
        n_fft = 4096
        hop = self.hop_length
        n_frames = 1 + (len(mono) - n_fft) // hop if len(mono) >= n_fft else 0
        if n_frames <= 0:
            return np.zeros((12, 0), dtype=np.float32)

        idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
        frames = mono[idx] * np.hanning(n_fft).astype(np.float32)
        mag = np.abs(np.fft.rfft(frames, axis=1))
        freqs = np.fft.rfftfreq(n_fft, 1.0 / self.sr)

        # 只取有音高意义的频段，避开直流与超高频噪声
        usable = (freqs >= 55.0) & (freqs <= 2000.0)
        f = freqs[usable]
        m = mag[:, usable]

        # 频率 → 相对 C 的音级（浮点，便于加权到相邻音级）
        pc = (12.0 * np.log2(np.maximum(f, 1e-6) / 440.0) + 69.0) % 12.0
        chroma = np.zeros((n_frames, 12), dtype=np.float32)
        lo = np.floor(pc).astype(np.int64) % 12
        hi = (lo + 1) % 12
        w_hi = (pc - np.floor(pc)).astype(np.float32)
        for b in range(m.shape[1]):
            chroma[:, lo[b]] += m[:, b] * (1.0 - w_hi[b])
            chroma[:, hi[b]] += m[:, b] * w_hi[b]

        # 每个音级做能量归一化（对数压缩），抑制响度差异
        chroma = np.log1p(chroma)
        return chroma.T

    # ---------- 匹配与平滑 ----------

    def _match(self, chroma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """逐帧匹配模板，返回 (分数, 模板索引)。"""
        if chroma.shape[1] == 0:
            return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.int64)
        x = chroma.T.astype(np.float32)               # (n_frames, 12)
        norm = np.linalg.norm(x, axis=1, keepdims=True)
        x = x / np.maximum(norm, 1e-9)
        # 余弦相似度：模板已归一化
        sim = x @ self._templates.T                   # (n_frames, n_templates)
        best = np.argmax(sim, axis=1)
        score = sim[np.arange(sim.shape[0]), best]
        return score.astype(np.float32), best.astype(np.int64)

    @staticmethod
    def _median_smooth(labels: np.ndarray, kernel: int = 5) -> np.ndarray:
        """对标签序列做中值滤波，去掉单帧跳变。"""
        if labels.size < kernel or kernel < 3:
            return labels
        half = kernel // 2
        out = labels.copy()
        for i in range(labels.size):
            lo = max(0, i - half)
            hi = min(labels.size, i + half + 1)
            vals, counts = np.unique(labels[lo:hi], return_counts=True)
            out[i] = vals[np.argmax(counts)]
        return out

    def _segments(self, labels: np.ndarray, scores: np.ndarray,
                  n_frames: int) -> list[ChordSegment]:
        """把逐帧标签合并成和弦段，应用最短持续时间约束。"""
        segments: list[ChordSegment] = []
        if labels.size == 0:
            return segments
        min_frames = max(1, int((self.min_duration_ms / 1000.0) / self.hop_sec))

        start = 0
        for i in range(1, labels.size + 1):
            if i == labels.size or labels[i] != labels[start]:
                length = i - start
                if length >= min_frames:
                    root, quality = self._labels[int(labels[start])]
                    conf = float(scores[start:i].mean())
                    if conf >= self.min_confidence:
                        pitches = self._voicing(root, quality)
                        segments.append(ChordSegment(
                            start=start * self.hop_sec,
                            end=i * self.hop_sec,
                            label=f"{_PITCH_CLASSES[root]}{quality}",
                            root=int(root), quality=quality,
                            confidence=conf, pitches=pitches,
                        ))
                start = i
        if self.merge_short_ms > 0 and len(segments) >= 3:
            segments = self._merge_short_between_same(segments)
        return segments

    def _merge_short_between_same(self, segments: list["ChordSegment"]
                                  ) -> list["ChordSegment"]:
        """把「A - B(很短) - A」中间那段 B 并入 A。

        短促的和弦切换绝大多数不是真的转调，而是琶音、装饰音、踏板混响
        导致的逐帧标签抖动。前后同和弦时，中间那段几乎必然是误判。
        """
        th = self.merge_short_ms / 1000.0
        changed = True
        while changed and len(segments) >= 3:
            changed = False
            for i in range(1, len(segments) - 1):
                prev, mid, nxt = segments[i - 1], segments[i], segments[i + 1]
                dur = mid.end - mid.start
                if dur > th:
                    continue
                if prev.label == nxt.label:
                    prev.end = nxt.end
                    prev.confidence = max(prev.confidence, nxt.confidence)
                    del segments[i + 1]
                    del segments[i]
                    changed = True
                    break
        return segments

    @staticmethod
    def _voicing(root: int, quality: str) -> list[int]:
        """把和弦模板转成具体 MIDI 音高，根音落在中音区。"""
        degrees = CHORD_TEMPLATES[quality][0]
        base = 12 * (CHORD_BASE_OCTAVE + 1)  # C4 = 60
        pitches = []
        for d in degrees:
            p = base + root + d
            # 超过一个八度就下移，保持和弦在紧凑音域内
            while p > base + 11:
                p -= 12
            pitches.append(int(p))
        return sorted(pitches)

    # ---------- 主流程 ----------

    def transcribe(self, audio, sr: int | None = None,
                   instrument: int = 1) -> StemNotes:
        from ..ingest import resample

        if isinstance(audio, str):
            import soundfile as sf

            data, file_sr = sf.read(audio, dtype="float32", always_2d=True)
            data = data.T
        else:
            data = np.asarray(audio, dtype=np.float32)
            if data.ndim == 1:
                data = data[None, :]
            file_sr = int(sr or self.sr)

        if file_sr != self.sr:
            data = resample(data, file_sr, self.sr)
        mono = data.mean(axis=0) if data.shape[0] > 1 else data.reshape(-1)

        if not self.enabled or mono.size < 4096:
            return StemNotes(stem="chords", notes=[], instrument=instrument,
                             meta={"engine": "chord_template", "enabled": False})

        chroma = self._chroma(mono)
        if chroma.shape[1] == 0:
            return StemNotes(stem="chords", notes=[], instrument=instrument,
                             meta={"engine": "chord_template", "reason": "chroma 为空"})

        scores, best = self._match(chroma)
        best = self._median_smooth(best, kernel=5)
        segments = self._segments(best, scores, chroma.shape[1])

        notes: list[NoteEvent] = []
        for seg in segments:
            # 每个和弦音给一个略短于段长的时值，留出换和弦的间隙，
            # 避免相邻和弦在听感上糊成一片。
            dur = max(0.15, min(seg.end - seg.start, 4.0)) * 0.92
            vel = int(np.clip(45 + 55 * seg.confidence, 40, 110))
            for p in seg.pitches:
                notes.append(NoteEvent(start=seg.start, end=seg.start + dur,
                                       pitch=p, velocity=vel))

        result = StemNotes(stem="chords", notes=notes, instrument=instrument)
        hist: dict[str, int] = {}
        for s in segments:
            hist[s.label] = hist.get(s.label, 0) + 1
        result.meta = {
            "engine": "chord_template",
            "hop_sec": self.hop_sec,
            "min_confidence": self.min_confidence,
            "min_duration_ms": self.min_duration_ms,
            "merge_short_ms": self.merge_short_ms,
            "segments": len(segments),
            "unique_chords": len(hist),
            "chord_histogram": dict(sorted(hist.items(), key=lambda kv: -kv[1])[:12]),
            "chroma_frames": int(chroma.shape[1]),
            "segments_detail": [s.to_dict() for s in segments[:200]],
        }
        return result
