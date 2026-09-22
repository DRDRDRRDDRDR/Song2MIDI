"""后处理：节拍估计、量化、清理。

输出 MIDI 的可用度，很大程度取决于这一层，而不是模型本身。
模型给出的是「音符在时间轴上的近似位置」，要变成能在 DAW 里对齐、
能循环、能继续编辑的 MIDI，必须解决三个问题：

    1. 时间位置是连续的浮点，而音乐是离散网格 → 需要节拍对齐与量化。
    2. 存在大量几十毫秒的碎片音符与同音高重复 → 需要合并与过滤。
    3. 力度分布不稳定 → 需要归一化。

节拍估计为何自研：量化必须对齐到真实节拍相位，而不是从 0 秒起算的
绝对网格。若只估 BPM 不估相位，整轨会有一处固定的系统性偏移，
听感上「整体抢拍或拖拍」，这是很常见且很难事后察觉的问题。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .config import Config
from .notes import NoteEvent, StemNotes, clamp_velocity

__all__ = ["PostProcessor", "TempoInfo", "estimate_tempo"]

# 有效 BPM 搜索范围
BPM_MIN = 55.0
BPM_MAX = 200.0

# 节拍感知先验：以该速度为对数正态中心，抑制倍速/半速错误。
# 依据是人对 90~180 BPM 的节拍最为敏感，而自相关在半速与倍速处往往同样强。
TEMPO_PRIOR_CENTER = 120.0
TEMPO_PRIOR_SIGMA = 0.75

# 节拍包络的帧率，与音高转录的 ANNOTATIONS_FPS 对齐
ENVELOPE_FPS = 86.0

_QUANTIZE_DIVISIONS = {
    "off": 0,
    "1/4": 1,
    "1/8": 2,
    "1/8t": 3,
    "1/16": 4,
    "1/16t": 6,
    "1/32": 8,
}


@dataclass
class TempoInfo:
    bpm: float
    phase_sec: float        # 第一节拍相对于 0 秒的偏移
    confidence: float
    source: str             # auto | config

    def to_dict(self) -> dict[str, Any]:
        return {"bpm": round(self.bpm, 3), "phase_sec": round(self.phase_sec, 4),
                "confidence": round(self.confidence, 3), "source": self.source}


def _onset_envelope(mono: np.ndarray, sr: int, fps: float = ENVELOPE_FPS) -> np.ndarray:
    """计算节拍包络：短时幅度谱的正向通量，再做半波整流。"""
    hop = max(1, int(round(sr / fps)))
    n_fft = 2048
    if len(mono) < n_fft:
        return np.zeros(0, dtype=np.float32)
    n_frames = 1 + (len(mono) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = mono[idx] * np.hanning(n_fft).astype(np.float32)
    mag = np.abs(np.fft.rfft(frames, axis=1))
    # 对数压缩，让弱拍也能贡献信息
    mag = np.log1p(mag)
    diff = np.diff(mag, axis=0)
    env = np.maximum(diff, 0.0).sum(axis=1)
    if env.size:
        env = env - env.mean()
    return env.astype(np.float32)


def estimate_tempo(mono: np.ndarray, sr: int,
                   bpm_min: float = BPM_MIN, bpm_max: float = BPM_MAX,
                   fps: float = ENVELOPE_FPS) -> TempoInfo:
    """用自相关估计 BPM，再用脉冲串互相关估计节拍相位。"""
    env = _onset_envelope(mono, sr, fps)
    if env.size < int(2 * fps):
        return TempoInfo(bpm=120.0, phase_sec=0.0, confidence=0.0, source="fallback")

    env = env / max(float(env.std()), 1e-9)

    # ---- BPM：自相关峰值 × 感知先验 ----
    #
    # 不能只取自相关最大峰。节拍的自相关在周期与 2 倍周期处往往同样强，
    # 直接取最大值会系统性落到半速或倍速上 —— 实测某曲目估出 63.7 BPM，
    # 而真实速度约 127 BPM，两者是 1:2 的关系。
    # 这类错误代价很大：量化网格会整体差一倍（1/16 在 63.7 BPM 下是 235ms，
    # 在 127 BPM 下是 118ms），所有音符时值都被拉到错误的格点上。
    #
    # 做法与 librosa 的 tempogram 一致：给候选速度乘一个以 120 BPM 为中心的
    # 对数正态权重。人对 90~180 BPM 的节拍最敏感，倍速/半速的候选会被压制。
    ac_full = np.correlate(env, env, mode="full")[env.size - 1:]
    ac_full = ac_full / max(ac_full[0], 1e-9)
    lag_min = max(1, int(round(60.0 / bpm_max * fps)))
    lag_max = min(ac_full.size - 1, int(round(60.0 / bpm_min * fps)))
    if lag_max <= lag_min:
        return TempoInfo(bpm=120.0, phase_sec=0.0, confidence=0.0, source="fallback")

    lags = np.arange(lag_min, lag_max + 1)
    cand = ac_full[lag_min:lag_max + 1].astype(np.float64)
    bpms = 60.0 * fps / lags
    prior = np.exp(-0.5 * (np.log(bpms / TEMPO_PRIOR_CENTER) / TEMPO_PRIOR_SIGMA) ** 2)
    score = cand * prior

    best_i = int(np.argmax(score))
    best_lag = int(lags[best_i])
    bpm = float(bpms[best_i])
    peak = float(cand[best_i])
    prior_weight = float(prior[best_i])

    # ---- 相位：用脉冲串与包络做互相关 ----
    period_frames = best_lag
    phase_scores = np.zeros(period_frames, dtype=np.float32)
    for p in range(period_frames):
        train = np.zeros_like(env)
        train[p::period_frames] = 1.0
        phase_scores[p] = float(np.dot(env, train))
    best_phase = int(np.argmax(phase_scores))
    phase_sec = best_phase / fps

    # 置信度同时反映自相关强度与先验接纳程度，
    # 若靠先验强行扭转了倍速关系，置信度应当相应下降而不是虚高。
    conf = float(np.clip(peak * (0.5 + 0.5 * prior_weight), 0.0, 1.0))
    return TempoInfo(bpm=float(bpm), phase_sec=float(phase_sec),
                     confidence=conf, source="auto")


class PostProcessor:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        sec = cfg.section("postprocess")
        self.quantize = str(sec.get("quantize", "1/16")).lower()
        self.bpm_cfg = sec.get("bpm", "auto")
        self.min_note_ms = float(sec.get("min_note_ms", 50.0))
        self.merge_gap_ms = float(sec.get("merge_gap_ms", 40.0))
        self.merge_tol = int(sec.get("merge_tolerance_semitones", 0))
        self.remove_out_of_range = bool(sec.get("remove_out_of_range", True))
        self.normalize_velocity = bool(sec.get("velocity_from_energy", True))
        self.pitch_lo = int(cfg.get("postprocess.pitch_min", 21))
        self.pitch_hi = int(cfg.get("postprocess.pitch_max", 108))

    # ---------- 各处理步骤 ----------

    @staticmethod
    def drop_short(notes: Iterable[NoteEvent], min_ms: float) -> tuple[list[NoteEvent], int]:
        keep = [n for n in notes if n.duration * 1000.0 >= min_ms]
        return keep, len(list(notes)) - len(keep)

    def merge_adjacent(self, notes: Iterable[NoteEvent]) -> tuple[list[NoteEvent], int]:
        """合并同音高、间隔很小的相邻音符。

        分离模型常把一个持续音切成若干短音，听感上会变成「打嗝」。
        合并后既更接近真实演奏，也大幅减少 MIDI 事件数。
        """
        items = sorted(notes, key=lambda n: (n.pitch, n.start))
        merged: list[NoteEvent] = []
        gap_sec = self.merge_gap_ms / 1000.0
        count = 0

        for n in items:
            if not merged:
                merged.append(NoteEvent(n.start, n.end, n.pitch, n.velocity,
                                        list(n.pitch_bends) if n.pitch_bends else None))
                continue
            prev = merged[-1]
            same_pitch = abs(n.pitch - prev.pitch) <= self.merge_tol
            # 允许轻微重叠（模型输出的相邻音常有 10ms 级重叠）
            close = n.start - prev.end <= gap_sec
            if same_pitch and close:
                prev.end = max(prev.end, n.end)
                # 力度取较大者：保留起音强度，避免合并后整体变弱
                prev.velocity = max(prev.velocity, n.velocity)
                count += 1
            else:
                merged.append(NoteEvent(n.start, n.end, n.pitch, n.velocity,
                                        list(n.pitch_bends) if n.pitch_bends else None))
        return merged, count

    # ---------- 伪影清理 ----------
    #
    # 这一段针对「模型幻觉出的、原曲没有的音符」。四条规则都有物理依据，
    # 不是拍脑袋定的阈值：
    #   1. 极短音      —— 钢琴最短可辨识音约 30~50ms，比这更短的不可能是真实击键
    #   2. 低力度短音  —— 力度对应击键强度，真钢琴不会既极弱又极短
    #   3. 同音极短重复 —— 人手无法在 50ms 内重复按下同一个键
    #   4. 复音上限    —— 同时发声数超过设定值（钢琴十指 + 踏板可放宽）
    #
    # 全部默认关闭（阈值 0 = 不启用），避免在用户不知情时改动结果。

    def cleanup_artifacts(self, notes: Iterable[NoteEvent],
                          ultra_short_ms: float = 0.0,
                          weak_below: int = 0,
                          weak_max_ms: float = 0.0,
                          dedupe_repeat_ms: float = 0.0,
                          ) -> tuple[list[NoteEvent], dict[str, int]]:
        """去除伪影音符。返回 (保留的音符, 各项删除计数)。"""
        items = list(notes)
        stats = {"ultra_short": 0, "weak_short": 0, "repeat": 0}

        # 1) 极短音
        if ultra_short_ms > 0:
            th = ultra_short_ms / 1000.0
            kept = []
            for n in items:
                if n.duration < th:
                    stats["ultra_short"] += 1
                else:
                    kept.append(n)
            items = kept

        # 2) 低力度 + 短时值
        if weak_below > 0 and weak_max_ms > 0:
            th = weak_max_ms / 1000.0
            kept = []
            for n in items:
                if n.velocity < weak_below and n.duration < th:
                    stats["weak_short"] += 1
                else:
                    kept.append(n)
            items = kept

        # 3) 同音高极短重复去重：间隔 < 阈值且重叠的，只保留较长者
        if dedupe_repeat_ms > 0:
            th = dedupe_repeat_ms / 1000.0
            by_pitch: dict[int, list[NoteEvent]] = {}
            for n in items:
                by_pitch.setdefault(n.pitch, []).append(n)
            drop: set[int] = set()
            for pitch, group in by_pitch.items():
                group.sort(key=lambda x: x.start)
                for i in range(1, len(group)):
                    a, b = group[i - 1], group[i]
                    if id(a) in drop:
                        continue
                    if b.start - a.end <= th:
                        # 保留时长更大者
                        if b.duration > a.duration:
                            drop.add(id(a))
                        else:
                            drop.add(id(b))
                            stats["repeat"] += 1
            items = [n for n in items if id(n) not in drop]

        return items, stats

    def limit_polyphony(self, notes: Iterable[NoteEvent],
                        max_poly: int = 0) -> tuple[list[NoteEvent], int]:
        """同时发声数超限时，删掉时值较短者。

        钢琴上要谨慎：**踏板延音下同时发声音符本就可以超过十指**，
        所以这个上限应设得比 10 宽（如 16），它针对的是明显的伪影堆积。
        """
        if max_poly <= 0:
            return list(notes), 0
        items = list(notes)
        bounds = sorted({n.start for n in items} | {n.end for n in items})
        removed: set[int] = set()
        for t0, t1 in zip(bounds, bounds[1:]):
            if t1 <= t0:
                continue
            mid = (t0 + t1) / 2.0
            active = [i for i, n in enumerate(items)
                      if i not in removed and n.start <= mid < n.end]
            if len(active) <= max_poly:
                continue
            active.sort(key=lambda i: (-items[i].duration, -items[i].velocity))
            for i in active[max_poly:]:
                removed.add(i)
        return [n for i, n in enumerate(items) if i not in removed], len(removed)

    def fold_range(self, notes: Iterable[NoteEvent]) -> tuple[list[NoteEvent], int]:
        """把超出音域的音符以八度平移折叠回范围内。

        选择折叠而不是直接删除：越界音符通常是八度检测偏差，
        折回后音级信息仍是对的，信息不丢；直接删会凭空少掉声部。
        """
        out: list[NoteEvent] = []
        folded = dropped = 0
        for n in notes:
            p = n.pitch
            if self.pitch_lo <= p <= self.pitch_hi:
                out.append(n)
                continue
            shifted = p
            while shifted < self.pitch_lo:
                shifted += 12
            while shifted > self.pitch_hi:
                shifted -= 12
            if self.pitch_lo <= shifted <= self.pitch_hi:
                out.append(n.shifted(shifted - p))
                folded += 1
            else:
                dropped += 1
        return out, folded + dropped

    def resolve_overlaps(self, notes: Iterable[NoteEvent]) -> list[NoteEvent]:
        """同音高音符不允许时间重叠。

        MIDI 规范允许多次 note-on 同名音，但多数音源与 DAW 在遇到
        重叠的同音高音符时会出现 note-off 配对错乱，导致音符提前中断。
        """
        by_pitch: dict[int, list[NoteEvent]] = {}
        for n in notes:
            by_pitch.setdefault(n.pitch, []).append(n)

        out: list[NoteEvent] = []
        for pitch, group in by_pitch.items():
            group.sort(key=lambda x: x.start)
            for i, n in enumerate(group):
                if i + 1 < len(group) and n.end > group[i + 1].start:
                    n = NoteEvent(n.start, max(n.start + 0.01, group[i + 1].start),
                                  n.pitch, n.velocity, n.pitch_bends)
                out.append(n)
        return out

    @staticmethod
    def stretch_velocity(notes: list[NoteEvent]) -> list[NoteEvent]:
        """把力度拉伸到较宽的动态范围。

        模型输出的力度整体偏暗（常见均值 50 上下），直接写进 MIDI
        听起来会「没有起伏」。按分位数做线性拉伸，保留相对关系的同时
        让动态更接近真实演奏。

        命名注意：不要叫 normalize_velocity —— 那会和开关属性
        self.normalize_velocity 重名，属性会遮蔽方法，调用时报
        「'bool' object is not callable」。这个坑踩过一次。
        """
        if len(notes) < 8:
            return notes
        v = np.array([n.velocity for n in notes], dtype=np.float32)
        lo, hi = np.percentile(v, 5), np.percentile(v, 95)
        if hi - lo < 5:
            return notes
        scaled = np.clip(45 + (v - lo) / (hi - lo) * 75, 25, 120)
        for n, s in zip(notes, scaled):
            n.velocity = clamp_velocity(s)
        return notes

    def quantize_notes(self, notes: list[NoteEvent], tempo: TempoInfo
                       ) -> tuple[list[NoteEvent], int]:
        """把音符起止量化到节拍网格。"""
        div = _QUANTIZE_DIVISIONS.get(self.quantize, 0)
        if div == 0 or tempo.bpm <= 0:
            return notes, 0

        step = (60.0 / tempo.bpm) / div       # 网格步长（秒）
        if step <= 0:
            return notes, 0
        phase = tempo.phase_sec
        min_len = max(0.02, self.min_note_ms / 1000.0)
        changed = 0

        for n in notes:
            s = round((n.start - phase) / step) * step + phase
            e = round((n.end - phase) / step) * step + phase
            if e <= s:
                e = s + min_len
            if abs(s - n.start) > 1e-6 or abs(e - n.end) > 1e-6:
                changed += 1
            n.start = max(0.0, float(s))
            n.end = float(e)
        return notes, changed

    # ---------- 主流程 ----------

    def process(self, stem: StemNotes, tempo: TempoInfo | None = None,
                quantize: bool | None = None) -> StemNotes:
        notes = list(stem.notes)
        stats: dict[str, Any] = {"in": len(notes)}

        # 鼓组不参与量化起音之外的音高折叠（鼓音高是乐器编号，不是音高）
        if stem.is_drum_track:
            notes, dropped = self.drop_short(notes, max(20.0, self.min_note_ms / 2))
            stats["dropped_short"] = dropped
            if self.normalize_velocity:
                notes = self.stretch_velocity(notes)
            if quantize is not False and tempo is not None:
                notes, q = self.quantize_notes(notes, tempo)
                stats["quantized"] = q
            result = StemNotes(stem.stem, notes, stem.instrument, stem.is_drum_track)
            result.meta = {**stem.meta, "postprocess": stats}
            return result

        notes, dropped = self.drop_short(notes, self.min_note_ms)
        stats["dropped_short"] = dropped

        notes, merged = self.merge_adjacent(notes)
        stats["merged"] = merged

        # ---- 伪影清理（可开关，阈值 0 = 跳过）----
        #
        # 位置很关键：**必须放在 merge_adjacent 之后**。
        # 清理会把「同一个音被切成的碎片」中的一部分删掉，若放在 merge 之前，
        # 剩下的碎片不再相邻（间隔被拉大超过 merge 阈值），merge 链被打断，
        # 最终音符数反而**增加** —— 实测放前面会让音符数 +18.2%，完全反效果。
        # 放到 merge 之后，碎片已拼回完整音，此时的判定对象才是真正的音符。
        clean_cfg = self.cfg.section("postprocess").get("cleanup") or {}
        if clean_cfg.get("enabled"):
            notes, clean_stats = self.cleanup_artifacts(
                notes,
                ultra_short_ms=float(clean_cfg.get("drop_shorter_than_ms", 0) or 0),
                weak_below=int(clean_cfg.get("drop_weak_below", 0) or 0),
                weak_max_ms=float(clean_cfg.get("weak_max_ms", 0) or 0),
                dedupe_repeat_ms=float(clean_cfg.get("dedupe_repeat_ms", 0) or 0),
            )
            stats["cleanup"] = clean_stats

        if self.remove_out_of_range:
            notes, folded = self.fold_range(notes)
            stats["range_adjusted"] = folded
        stats["range"] = [self.pitch_lo, self.pitch_hi]

        notes = self.resolve_overlaps(notes)

        if clean_cfg.get("enabled"):
            notes, poly_dropped = self.limit_polyphony(
                notes, max_poly=int(clean_cfg.get("max_polyphony", 0) or 0))
            if poly_dropped:
                stats["polyphony_dropped"] = poly_dropped

        if quantize is not False and tempo is not None:
            notes, q = self.quantize_notes(notes, tempo)
            stats["quantized"] = q
            stats["quantize_grid"] = self.quantize
            stats["bpm"] = round(tempo.bpm, 3)
            stats["phase_sec"] = round(tempo.phase_sec, 4)

        if self.normalize_velocity:
            notes = self.stretch_velocity(notes)

        notes.sort(key=lambda n: (n.start, n.pitch))
        stats["out"] = len(notes)

        # 用关键字构造并**带上 pedals**：后处理会重建 StemNotes，
        # 漏传踏板会把钢琴引擎刚检出的踏板整段丢掉。
        # 另外踏板不参与量化 —— 它是连续控制，强行对齐到网格只会让它失真。
        result = StemNotes(stem=stem.stem, notes=notes, instrument=stem.instrument,
                           is_drum_track=stem.is_drum_track,
                           pedals=list(getattr(stem, "pedals", []) or []))
        result.meta = {**stem.meta, "postprocess": stats}
        return result

    def resolve_tempo(self, mono: np.ndarray, sr: int) -> TempoInfo:
        """按配置决定用固定 BPM 还是自动估计。"""
        if isinstance(self.bpm_cfg, (int, float)) and float(self.bpm_cfg) > 0:
            return TempoInfo(bpm=float(self.bpm_cfg), phase_sec=0.0,
                             confidence=1.0, source="config")
        try:
            bpm_val = float(self.bpm_cfg)
            if bpm_val > 0:
                return TempoInfo(bpm=bpm_val, phase_sec=0.0, confidence=1.0, source="config")
        except (TypeError, ValueError):
            pass
        return estimate_tempo(mono, sr)
