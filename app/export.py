"""多轨 MIDI 与音符数据导出。

输出三种形态：
    <曲名>_multitrack.mid   所有分轨在一个文件里，各自独立音轨，便于整体试听
    <曲名>_<分轨>.mid       每个分轨单独一个文件，便于在 DAW 里只取需要的部分
    <曲名>_notes.json       全部音符与元信息，便于排查与二次加工

关于鼓轨：General MIDI 约定打击乐固定使用第 10 通道（0 基索引为 9），
通道号由 pretty_midi 依据 Instrument.is_drum 自动分配，因此这里只需正确
设置 is_drum 标志，不要手动指定通道。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .config import Config
from .notes import NoteEvent, StemNotes, pitch_name

__all__ = ["MidiExporter", "ExportResult"]

# 各分轨默认 GM 音色（仅作兜底，正常走 config.yaml 的 program_map）
DEFAULT_PROGRAMS = {
    "vocals": 53,   # Voice Oohs
    "bass": 33,     # Electric Bass (finger)
    "other": 1,     # Acoustic Grand Piano
    "drums": 0,     # 通道 10 鼓组
    "chords": 1,    # Acoustic Grand Piano
    "mix": 1,
}

# 音轨名（写进 MIDI 的 track name，在 DAW 里能看到）
TRACK_LABELS = {
    "vocals": "Vocals",
    "drums": "Drums",
    "bass": "Bass",
    "other": "Other (Guitar/Keys/Synth)",
    "chords": "Chords",
    "mix": "Mix",
}


class ExportResult:
    def __init__(self) -> None:
        self.midi_files: list[str] = []
        self.json_file: str | None = None
        self.track_summary: list[dict[str, Any]] = []
        self.warnings: list[str] = []

    def to_dict(self) -> dict[str, Any]:
        return {
            "midi_files": self.midi_files,
            "json_file": self.json_file,
            "tracks": self.track_summary,
            "warnings": self.warnings,
        }


class MidiExporter:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.out_dir = cfg.path("paths.out_dir")
        self.midi_type = str(cfg.get("output.midi_type", "multi_track"))
        self.also_json = bool(cfg.get("output.also_write_json", True))
        self.program_map = dict(cfg.get("output.program_map") or {})

    # ---------- 辅助 ----------

    def _program(self, stem: str, fallback: int) -> int:
        val = self.program_map.get(stem)
        if val is None:
            return DEFAULT_PROGRAMS.get(stem, fallback)
        try:
            return max(0, min(127, int(val)))
        except (TypeError, ValueError):
            return DEFAULT_PROGRAMS.get(stem, fallback)

    @staticmethod
    def _add_notes(instrument, notes: Iterable[NoteEvent]) -> int:
        import pretty_midi

        n = 0
        for note in notes:
            # 跳过非法或零长音符，避免写出的 MIDI 在部分音源上卡音
            if note.end <= note.start:
                continue
            pitch = int(note.pitch)
            if not (0 <= pitch <= 127):
                continue
            instrument.notes.append(pretty_midi.Note(
                velocity=int(max(1, min(127, note.velocity))),
                pitch=pitch,
                start=float(max(0.0, note.start)),
                end=float(note.end),
            ))
            n += 1
        return n

    @staticmethod
    def _add_pedals(instrument, pedals) -> int:
        """把延音踏板区间展开成 MIDI CC64 事件。

        钢琴专用引擎输出的是「踏板区间」（踩着的一段），而 MIDI 里表达为两个
        控制变更：踩下 CC64=127、松开 CC64=0。

        为什么值得写：踏板是钢琴表现力的核心 —— 它让前一个音在手指离开后继续
        发声。只写音符不写踏板，钢琴曲听起来会明显偏干、偏断。
        """
        import pretty_midi

        n = 0
        for pd in pedals or ():
            start = float(max(0.0, getattr(pd, "start", 0.0)))
            end = float(getattr(pd, "end", 0.0))
            if end <= start:
                continue
            value = int(getattr(pd, "value", 127))
            value = max(1, min(127, value))
            # 踩下
            instrument.control_changes.append(pretty_midi.ControlChange(
                number=64, value=value, time=start))
            # 松开
            instrument.control_changes.append(pretty_midi.ControlChange(
                number=64, value=0, time=end))
            n += 1
        return n

    def _build(self, stems: list[StemNotes], bpm: float):
        """构建 PrettyMIDI 对象。"""
        import pretty_midi

        pm = pretty_midi.PrettyMIDI(initial_tempo=float(bpm or 120.0))
        summary: list[dict[str, Any]] = []

        for sn in stems:
            if not sn.notes:
                summary.append({"stem": sn.stem, "notes": 0, "skipped": "无音符"})
                continue
            program = self._program(sn.stem, sn.instrument)
            inst = pretty_midi.Instrument(
                program=program,
                is_drum=bool(sn.is_drum_track),
                name=TRACK_LABELS.get(sn.stem, sn.stem),
            )
            added = self._add_notes(inst, sn.sorted_notes())
            if added == 0:
                summary.append({"stem": sn.stem, "notes": 0, "skipped": "全部音符非法"})
                continue
            # 踏板事件写在同一乐器轨上（与 CC 的语义一致：作用于该通道）
            added_pedals = self._add_pedals(inst, getattr(sn, "pedals", None))
            pm.instruments.append(inst)

            pitches = [n.pitch for n in sn.notes]
            summary.append({
                "stem": sn.stem,
                "label": TRACK_LABELS.get(sn.stem, sn.stem),
                "notes": added,
                "pedals": added_pedals,
                "program": program,
                "is_drum": bool(sn.is_drum_track),
                "pitch_range": f"{pitch_name(min(pitches))}–{pitch_name(max(pitches))}",
                "duration_sec": round(max(n.end for n in sn.notes), 2),
            })
        return pm, summary

    # ---------- 导出 ----------

    def export(self, stems: list[StemNotes], basename: str, bpm: float = 120.0,
               extra_meta: dict[str, Any] | None = None) -> ExportResult:
        result = ExportResult()
        self.out_dir.mkdir(parents=True, exist_ok=True)

        usable = [s for s in stems if s.notes]
        if not usable:
            result.warnings.append("所有分轨均无音符，未生成 MIDI")
            return result

        pm, summary = self._build(usable, bpm)
        result.track_summary = summary

        if self.midi_type in ("multi_track", "both"):
            dst = self.out_dir / f"{basename}_multitrack.mid"
            pm.write(str(dst))
            result.midi_files.append(str(dst))

        if self.midi_type in ("per_stem", "both"):
            for sn in usable:
                one, _ = self._build([sn], bpm)
                dst = self.out_dir / f"{basename}_{sn.stem}.mid"
                one.write(str(dst))
                result.midi_files.append(str(dst))

        if self.also_json:
            payload = {
                "source": basename,
                "tempo_bpm": round(float(bpm or 120.0), 3),
                "stem_count": len(usable),
                "stems": [s.to_dict() for s in usable],
            }
            if extra_meta:
                payload["pipeline"] = extra_meta
            dst = self.out_dir / f"{basename}_notes.json"
            dst.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            result.json_file = str(dst)

        return result

    # ---------- 诊断 ----------

    def verify(self, midi_path: str | Path) -> dict[str, Any]:
        """回读刚写出的 MIDI，确认文件确实可解析且内容完整。

        写文件成功不等于文件正确。回读校验能挡住「写出零音符文件」
        这类静默失败。
        """
        import pretty_midi

        try:
            pm = pretty_midi.PrettyMIDI(str(midi_path))
        except Exception as e:
            return {"ok": False, "error": f"{e.__class__.__name__}: {e}"}

        return {
            "ok": True,
            "file": str(midi_path),
            "size_bytes": Path(midi_path).stat().st_size,
            "instruments": len(pm.instruments),
            "total_notes": sum(len(i.notes) for i in pm.instruments),
            "tempo_bpm": round(float(pm.get_tempo_changes()[1][0]), 3)
            if len(pm.get_tempo_changes()[1]) else None,
            "duration_sec": round(float(pm.get_end_time()), 2),
            "tracks": [
                {
                    "name": i.name,
                    # 必须显式转成原生类型：pretty_midi 的 program 是
                    # 从 MIDI 字节里读出的 numpy.int64，直接放进响应会让
                    # JSON 序列化失败（实测导致任务查询接口 500）。
                    "program": int(i.program),
                    "is_drum": bool(i.is_drum),
                    "notes": int(len(i.notes)),
                }
                for i in pm.instruments
            ],
        }
