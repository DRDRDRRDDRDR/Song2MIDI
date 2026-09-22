"""统一音符数据结构。

各引擎（音高转录 / 鼓组 / 和弦）内部算法差异极大，但对外必须收敛成同一种
表示，上层才能不加区分地量化、清理、写 MIDI。这个模块就是那个收敛点。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Iterable

__all__ = ["NoteEvent", "PedalEvent", "StemNotes", "clamp_velocity", "pitch_name"]

# General MIDI 鼓组音高范围，用于区分「打击乐轨」与「旋律轨」
GM_DRUM_PITCH_MIN = 35
GM_DRUM_PITCH_MAX = 81

_PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def clamp_velocity(v: float) -> int:
    """把任意力度值夹到 MIDI 合法区间 1..127。"""
    try:
        iv = int(round(float(v)))
    except (TypeError, ValueError):
        return 100
    return max(1, min(127, iv))


def pitch_name(pitch: int) -> str:
    """MIDI 音高转音名，如 60 → C4。"""
    p = int(pitch)
    return f"{_PITCH_NAMES[p % 12]}{p // 12 - 1}"


@dataclass
class NoteEvent:
    """一个音符事件。时间单位为秒，与 MIDI 内部表示一致，便于直接写出。"""

    start: float
    end: float
    pitch: int
    velocity: int = 100
    # 音高弯曲序列（可选）。仅在需要保留滑音/颤音时填充，
    # 序列值语义与 MIDI pitch wheel 一致（0..16383，中心 8192）。
    pitch_bends: list[int] | None = None

    # ---- 派生属性 ----

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def name(self) -> str:
        return pitch_name(self.pitch)

    @property
    def is_drum(self) -> bool:
        return GM_DRUM_PITCH_MIN <= self.pitch <= GM_DRUM_PITCH_MAX

    # ---- 变换 ----

    def shifted(self, semitones: int) -> "NoteEvent":
        """整体移调。用于把音符折叠回目标音域。"""
        return NoteEvent(
            start=self.start, end=self.end,
            pitch=int(self.pitch) + int(semitones),
            velocity=self.velocity,
            pitch_bends=list(self.pitch_bends) if self.pitch_bends else None,
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["name"] = self.name
        d["duration"] = round(self.duration, 4)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "NoteEvent":
        return cls(
            start=float(d["start"]), end=float(d["end"]),
            pitch=int(d["pitch"]), velocity=clamp_velocity(d.get("velocity", 100)),
            pitch_bends=d.get("pitch_bends"),
        )


@dataclass
class PedalEvent:
    """延音踏板事件（MIDI CC64）。

    钢琴演奏的表现力很大程度来自踏板 —— 它让前一个音在手指离开后继续发声，
    构成连贯的乐句。只写音符、不写踏板，钢琴曲听起来会明显偏干、偏断。

    这里的表示是「踏板区间」（踩着的一段），写出 MIDI 时展开成
    CC64=127（踩下）与 CC64=0（松开）两个事件。
    """

    start: float
    end: float
    value: int = 127        # 127=踩下并保持；保留字段以便将来支持半踏

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": round(self.start, 4),
            "end": round(self.end, 4),
            "duration": round(self.duration, 4),
            "value": int(self.value),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PedalEvent":
        return cls(start=float(d["start"]), end=float(d["end"]),
                   value=int(d.get("value", 127)))


@dataclass
class StemNotes:
    """一路分轨的转录结果。

    注意字段顺序：`notes → instrument → is_drum_track → meta` 是原有顺序，
    不少地方按**位置参数**构造（如 postprocess 里的
    `StemNotes(stem.stem, notes, stem.instrument, stem.is_drum_track)`）。
    因此新增字段一律追加到**末尾**，插在中间会让位置参数错位 ——
    表现为 `instrument` 被当成别的字段，报出与真实原因无关的错误。
    """

    stem: str                       # vocals / drums / bass / other / chords / mix
    notes: list[NoteEvent] = field(default_factory=list)
    instrument: int = 1             # GM 音色编号
    is_drum_track: bool = False     # True 时写出 MIDI 需走通道 10
    meta: dict[str, Any] = field(default_factory=dict)
    # 延音踏板区间。目前只有钢琴专用引擎会填充，其他引擎留空。
    pedals: list[PedalEvent] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.notes)

    def add(self, note: NoteEvent) -> None:
        self.notes.append(note)

    def extend(self, notes: Iterable[NoteEvent]) -> None:
        self.notes.extend(notes)

    def sorted_notes(self) -> list[NoteEvent]:
        return sorted(self.notes, key=lambda n: (n.start, n.pitch))

    # ---- 统计，用于结果回显与调试 ----

    def stats(self) -> dict[str, Any]:
        if not self.notes:
            return {"stem": self.stem, "count": 0, "pedal_count": len(self.pedals)}
        pitches = [n.pitch for n in self.notes]
        durs = [n.duration for n in self.notes]
        ends = [n.end for n in self.notes]
        return {
            "stem": self.stem,
            "count": len(self.notes),
            "pedal_count": len(self.pedals),
            "pitch_min": min(pitches),
            "pitch_max": max(pitches),
            "pitch_min_name": pitch_name(min(pitches)),
            "pitch_max_name": pitch_name(max(pitches)),
            "duration_total_sec": round(sum(durs), 2),
            "duration_mean_ms": round(1000 * sum(durs) / len(durs), 1),
            "last_note_end_sec": round(max(ends), 2),
            "avg_velocity": round(sum(n.velocity for n in self.notes) / len(self.notes), 1),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "stem": self.stem,
            "instrument": self.instrument,
            "is_drum_track": self.is_drum_track,
            "meta": self.meta,
            "stats": self.stats(),
            "notes": [n.to_dict() for n in self.sorted_notes()],
            "pedals": [p.to_dict() for p in self.pedals],
        }
