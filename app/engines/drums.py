"""鼓组转录：onset 检测 + 频段特征分类 → GM 鼓组 MIDI。

为何自研而不用现成方案：
    madmom 是目前鼓组转录效果最好的开源库，但它依赖 Cython 扩展，
    没有 Python 3.13 的预编译 wheel，安装会失败；其余备选（librosa 自带
    的 onset 检测）只给时间点、不给鼓件类别，仍然需要自己做分类。
    因此这里直接实现「分频段频谱通量 + 规则分类」，零第三方依赖，
    行为完全可解释、可调参。

原理：
    不同鼓件的能量集中频段不同 —— 底鼓在 20~150Hz，军鼓主体在 150~1000Hz
    且伴随宽带噪声，踩镲/镲片在 6kHz 以上。用各频段的频谱通量（相邻帧
    幅度谱的正向增量）作为「该频段在此刻是否被激励」的度量，就能在 onset
    时刻直接读出鼓件类别，无需训练模型。

局限（需如实告知）：
    这是启发式方法，不是学习模型。在鼓组已被 Demucs 干净分离的前提下效果
    尚可；若鼓与贝斯、低音合成器混在一起，底鼓与低频贝斯的区分会显著退化。
    踩镲的开/闭判定依赖衰减时间，快速连续击打时容易误判。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..config import Config
from ..notes import NoteEvent, StemNotes, clamp_velocity

__all__ = ["DrumTranscriber", "DrumHit"]

# 分析用固定参数。44100/512 = 86.13 fps，与音高转录的 ANNOTATIONS_FPS 对齐，
# 便于后续在统一时间网格上做量化。
N_FFT = 2048
HOP = 512

# 分类所用的频段边界（Hz）
BAND_LOW = (20.0, 150.0)
BAND_MID = (150.0, 1000.0)
BAND_HIGH = (1000.0, 6000.0)
BAND_VHIGH = (6000.0, 16000.0)

# 各频段的代表频率，取几何中点。
# 用于计算「频段激活重心」—— 以激活占比为权重对各段代表频率加权平均。
# 之所以不用频谱重心：频谱重心按幅度谱对线性分布的 FFT bin 加权，
# 高频段 bin 数量远多于低频段，会把重心系统性地往上拉，不可用于判别鼓件。
BAND_CENTERS_HZ = {
    "low": math.sqrt(BAND_LOW[0] * BAND_LOW[1]),
    "mid": math.sqrt(BAND_MID[0] * BAND_MID[1]),
    "high": math.sqrt(BAND_HIGH[0] * BAND_HIGH[1]),
    "vhigh": math.sqrt(BAND_VHIGH[0] * BAND_VHIGH[1]),
}

# 击打后用于测量衰减的时间窗
DECAY_WINDOW_MS = 400.0


@dataclass
class DrumHit:
    """一次检测到的击打。"""

    time: float
    kind: str
    velocity: int
    confidence: float
    # 各频段归一化激活强度（相对该频段 98 分位），即分类所用的实际特征
    norm_low: float
    norm_mid: float
    norm_high: float
    norm_vhigh: float
    # 高频残余能量比：绝对值与相对本曲中位数的倍数。
    # 分类实际使用的是相对值 —— 绝对值在密集素材里恒高，无法判别。
    sustain_vhigh: float
    sustain_rel: float
    decay_low_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "time": round(self.time, 4), "kind": self.kind, "velocity": self.velocity,
            "confidence": round(self.confidence, 3),
            "norm": {k: round(v, 3) for k, v in {
                "low": self.norm_low, "mid": self.norm_mid,
                "high": self.norm_high, "vhigh": self.norm_vhigh}.items()},
            "sustain_rel": round(self.sustain_rel, 3),
            "decay_low_ms": round(self.decay_low_ms, 1),
        }


def _frame_signal(x: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    """把一维信号切成帧，返回 (n_frames, n_fft)。末尾不足一帧则丢弃。"""
    n_frames = 1 + (len(x) - n_fft) // hop if len(x) >= n_fft else 0
    if n_frames <= 0:
        return np.zeros((0, n_fft), dtype=np.float32)
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    return x[idx]


def _magnitude_spectrogram(x: np.ndarray, n_fft: int = N_FFT,
                           hop: int = HOP) -> np.ndarray:
    """手工实现幅度谱，避免依赖任何音频库的 STFT 接口。"""
    frames = _frame_signal(np.asarray(x, dtype=np.float32), n_fft, hop)
    if frames.shape[0] == 0:
        return np.zeros((0, n_fft // 2 + 1), dtype=np.float32)
    window = np.hanning(n_fft).astype(np.float32)
    spec = np.fft.rfft(frames * window, axis=1)
    return np.abs(spec).astype(np.float32)


class DrumTranscriber:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        sec = cfg.section("drums")
        self.mapping: dict[str, int] = dict(sec.get("mapping") or {})
        self.sensitivity = float(sec.get("onset_sensitivity", 0.5))
        self.kick_max_hz = float(sec.get("kick_max_hz", 120.0))
        self.hat_min_hz = float(sec.get("hat_min_hz", 5000.0))

        # 分类阈值。单位是「相对该频段 95 分位的归一化激活强度」，
        # 1.0 约等于一次满强度击打。可在 config.yaml 的 drums 段覆盖。
        self.th_kick = float(sec.get("th_kick", 0.32))
        self.th_snare_mid = float(sec.get("th_snare_mid", 0.42))
        self.th_snare_high = float(sec.get("th_snare_high", 0.22))
        # 军鼓的额外约束：高频激活须达到低频的该倍数，用于排除底鼓的宽带瞬态
        self.th_snare_high_ratio = float(sec.get("th_snare_high_ratio", 0.50))
        self.th_hat = float(sec.get("th_hat", 0.30))
        self.th_low_tom_min = float(sec.get("th_low_tom_min", 0.45))
        self.th_low_tom_max = float(sec.get("th_low_tom_max", 0.85))
        # 开/闭镲判据：必须**同时**满足绝对下限与相对倍数，缺一不可。
        #
        #   绝对下限：残余能量本身得有一定量级。稀疏素材里闭镲的绝对残余
        #     只有 0.02~0.08，若只比倍数，闭镲之间的正常波动就能让一部分
        #     超过中位数的 1.8 倍而被误判成开镲（实测误报 6 次）。
        #   相对倍数：残余须明显超过本曲常态。密集素材里高频内容连贯，
        #     绝对残余恒高，只看绝对值会把所有镲都判成开镲
        #     （实测 30 秒内开镲 135 / 闭镲 4，完全颠倒）。
        # 两条同时要求，稀疏与密集素材都能正确判别。
        self.th_open_hat_sustain_abs = float(sec.get("th_open_hat_sustain_abs", 0.15))
        self.th_open_hat_sustain_rel = float(sec.get("th_open_hat_sustain_rel", 1.8))
        # 镲片要求比开镲拖得更久：绝对与相对同时提高
        self.th_crash_sustain_abs = float(sec.get("th_crash_sustain_abs", 0.30))
        self.th_crash_sustain_rel = float(sec.get("th_crash_sustain_rel", 3.0))
        # 上一次检测统计出的残余能量基线，供诊断查看
        self._last_sustain_baseline = 0.0
        sr = int(cfg.get("audio.target_sr", 44100))
        self.sr = sr
        freqs = np.fft.rfftfreq(N_FFT, 1.0 / sr)
        self.freqs = freqs

        def band(lo: float, hi: float) -> np.ndarray:
            return (freqs >= lo) & (freqs < hi)

        self.mask_low = band(*BAND_LOW)
        self.mask_mid = band(*BAND_MID)
        self.mask_high = band(*BAND_HIGH)
        self.mask_vhigh = band(*BAND_VHIGH)
        self.mask_kick = band(20.0, self.kick_max_hz)
        self.mask_hat = band(self.hat_min_hz, 16000.0)

    # ---------- 频谱分析 ----------

    def _flux(self, mag: np.ndarray) -> np.ndarray:
        """分频段频谱通量，**按各频段 bin 数量做平均**。

        这里必须用均值而不是求和，原因很具体：
            低频段 20~150Hz 在 2048 点 FFT 下只有约 7 个 bin，
            超高频段 6k~16kHz 有约 460 个 bin。
        直接求和会让超高频段在数量级上碾压低频段，结果是底鼓与军鼓
        在 onset 探测和鼓件分类两个环节都被踩镲彻底掩盖
        —— 实测表现为「所有击打都被判成踩镲、底鼓完全检测不到」。
        按 bin 数归一后各频段才具备可比性。
        """
        if mag.shape[0] < 2:
            return np.zeros((0, 4), dtype=np.float32)
        pos = np.maximum(np.diff(mag, axis=0), 0.0)
        cols = []
        for mask in (self.mask_low, self.mask_mid, self.mask_high, self.mask_vhigh):
            n = max(1, int(mask.sum()))
            cols.append(pos[:, mask].sum(axis=1) / n)
        return np.stack(cols, axis=1).astype(np.float32)

    # 说明：这里原本有一个 _zscore_bands（按标准差归一）实现，已删除。
    # 原因是标准差会被该频段自己的主要成分抬高，而主要成分恰恰就是要检测的
    # 那个鼓件 —— 实测底鼓在低频段的 z 值只有 3.9，军鼓在中频段却有 7.0，
    # 底鼓被自己「太常见」惩罚了。现改用 _band_norm 的 98 分位归一。

    def _pick_peaks(self, x: np.ndarray, sensitivity: float) -> np.ndarray:
        """峰值拾取。sensitivity 越大越灵敏（阈值越低）。

        阈值用**分位数**而不是「均值 + k 倍标准差」：
        后者在本场景下会失效 —— onset 强度是四个频段 z 分数之和，
        而四个频段在击打时刻是高度相关的，相加后标准差被放大近 4 倍，
        导致阈值被抬到只有最强击打才能越过的程度（实测只检出 15/56 个）。
        分位数阈值不受分布形状影响，参数含义也更直观。
        """
        if x.size < 3:
            return np.array([], dtype=np.int64)

        # 局部极大值候选
        cand = (x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:])
        idx = np.flatnonzero(cand) + 1
        if idx.size == 0:
            return idx

        # sensitivity 0 → 取包络的 97 分位为阈；sensitivity 1 → 83 分位
        pct = 97.0 - 14.0 * float(np.clip(sensitivity, 0.0, 1.0))
        thr = float(np.percentile(x, pct))
        idx = idx[x[idx] >= thr]
        if idx.size == 0:
            return idx
        # 相对下限：低于最强峰 12% 的视为噪声残留。
        # 必须要有这道下限，因为分位数是尺度无关的 —— 整段音频都很安静时，
        # 分位数依然会切出一批「相对最高但仍很弱」的点，那些不是鼓点。
        idx = idx[x[idx] >= 0.12 * float(x[idx].max())]
        if idx.size == 0:
            return idx

        # 抑制过近的重复检测（20ms 内只保留最强的一个）
        min_gap = max(1, int(0.020 * self.sr / HOP))
        keep: list[int] = []
        for i in idx:
            if keep and i - keep[-1] < min_gap:
                if x[i] > x[keep[-1]]:
                    keep[-1] = int(i)
            else:
                keep.append(int(i))
        return np.asarray(keep, dtype=np.int64)

    def _decay_ms(self, mag: np.ndarray, frame: int, mask: np.ndarray) -> float:
        """测击打后该频段能量衰减到峰值 1/e 所需时间（毫秒）。

        起点必须取窗口内的**能量峰值**，不能取 onset 帧本身：
        onset 是由帧间差分定义的，落差最大的那一帧往往还在起音之前，
        此时能量尚未冲到峰值，直接判「已衰减到 1/e 以下」会一律返回 0ms，
        结果是所有镲片都被判成闭镲。
        """
        n_decay = max(2, int(DECAY_WINDOW_MS / 1000.0 * self.sr / HOP) + 2)
        seg = mag[frame:frame + n_decay, :]
        if seg.shape[0] < 2:
            return 0.0
        seg = seg[:, mask]
        env = seg.sum(axis=1)
        if env.size < 2:
            return 0.0
        peak_idx = int(np.argmax(env))
        peak = float(env[peak_idx])
        if peak <= 0:
            return 0.0
        tail = env[peak_idx:]
        below = np.flatnonzero(tail <= peak / np.e)
        if below.size == 0:
            return DECAY_WINDOW_MS
        return float(below[0] * HOP / self.sr * 1000.0)

    def _sustain_ratio(self, mag: np.ndarray, frame: int, mask: np.ndarray) -> float:
        """击打后 70~163ms 区间内该频段的残余能量，相对起始峰值。

        用来区分开镲与闭镲。相比「1/e 衰减时间」这个判据更稳健：
        短时傅里叶的窗长（本实现为 2048 点约 46ms）会把瞬态在时间轴上抹开，
        一个真实衰减 17ms 的闭镲测出来的 1/e 时间可能长达 60ms 以上，
        与开镲的 111ms 出现重叠区，阈值怎么调都会顾此失彼。
        残余能量比衡量的是「过了一段时间还剩多少」，不依赖衰减曲线的形状，
        闭镲残余接近零，开镲仍有可观残响，分离度明显更好。

        取 70ms 起而非紧接起点，是为了避开窗长本身带来的抹开效应；
        取到 163ms 截止，是为了不碰到下一个八分音符（约 250ms 处）的起音。
        """
        f0 = max(1, int(0.070 * self.sr / HOP))
        f1 = max(f0 + 2, int(0.163 * self.sr / HOP))
        seg = mag[frame:frame + f1, :]
        if seg.shape[0] < 4:
            return 0.0
        env = seg[:, mask].sum(axis=1)
        if env.size <= f0 + 1:
            return 0.0
        early = float(env[:max(2, int(0.046 * self.sr / HOP))].max())
        if early <= 0:
            return 0.0
        tail = env[f0:f1]
        return float(tail.mean() / early) if tail.size else 0.0

    # 已被放弃的两个特征，记录在此以免后人重复踩坑：
    #
    # 1. 频谱重心（spectral centroid）
    #    它按幅度对线性分布的 FFT bin 加权，而高频段 bin 数远多于低频段
    #    （6k~16kHz 约 460 个 vs 20~150Hz 约 7 个），会把重心系统性拉高。
    #    实测底鼓与踩镲同拍时重心算出 7172Hz，而低频激活占比其实有 0.56。
    #
    # 2. 1/e 衰减时间（仍保留在 _decay_ms 中，供通鼓判别使用）
    #    窗长（2048 点约 46ms）会把瞬态在时间上抹开，导致闭镲测得 60ms 以上、
    #    与开镲的 111ms 出现重叠区，阈值无论怎么调都会顾此失彼。
    #    开/闭镲改判用 _sustain_ratio（残余能量比），分离度明显更好。

    # ---------- 分类 ----------

    @staticmethod
    def _band_norm(flux: np.ndarray) -> np.ndarray:
        """按各频段自身的 98 分位归一，把「一次强击打」映射到 1.0 附近。

        为何不用标准差归一（z 分数）：标准差会被该频段自己的主要成分抬高，
        而主要成分恰恰就是我们要检测的那个鼓件。实测底鼓在低频段的
        z 值只有 3.9，而军鼓在中频段是 7.0 —— 底鼓被自己「太常见」惩罚了。

        为何不用频段占比（各段换算成份额）：同拍多鼓同时击打时每个频段
        占比上限只有约 1/N，任何「占比占优」阈值都必然漏判。多标签判定
        应当逐频段独立比较自身的激活强度，而不是比较彼此的份额。

        取 98 分位而非 95 分位：鼓点通量分布极尖（多数帧接近零），
        95 分位落在「中等偏弱」的击打上，会让强击打归一后达到 5 以上，
        阈值失去直观含义。98 分位更接近「强击打」这一参考点。
        """
        if flux.shape[0] == 0:
            return flux
        p = np.percentile(flux, 98, axis=0)
        return (flux / (p + 1e-9)).astype(np.float32)

    def _classify_multi(self, norm: dict[str, float], sustain_abs: float,
                        sustain_rel: float, decay_low_ms: float
                        ) -> list[tuple[str, float]]:
        """判定该 onset 时刻有哪些鼓件被激励，返回 [(类别, 置信度), ...]。

        每个鼓件各自按「对应频段是否被显著激活」独立判定，互不排斥，
        因此同拍多鼓能正确还原成多个音符 —— 真实鼓点里底鼓与踩镲同拍
        极其常见，取单个最高分的做法会结构性丢失音符。

        三个由实测数据得出的判别约束：

        1. 军鼓除要求中频激活外，还要求高频相对低频达到一定比例。
           任何尖锐瞬态本质都是宽带的，底鼓起音同样会激励中频与高频，
           单看「高频被激活」无法与军鼓区分。实测纯底鼓帧的
           norm_high/norm_low 约 0.34，而底鼓+军鼓帧约 1.5，比值可分离。

        2. 开镲与闭镲用**相对本曲中位数**的高频残余能量比判别，不用绝对值。
           真实音乐的镲群密集且带残响，绝对阈值会把所有镲都判成开镲
           （实测 30 秒内开镲 135 / 闭镲 4，完全颠倒）。与中位数对比后，
           「明显比本曲常态更拖长」才是开镲。

        3. 通鼓用低频衰减时间判别，该处衰减差异大，1/e 判据仍然可靠。

        阈值均可在 config.yaml 的 drums 段调整（th_* 项）。
        """
        low = norm["low"]
        mid = norm["mid"]
        high = norm["high"]
        vhigh = norm["vhigh"]

        out: list[tuple[str, float]] = []

        # 底鼓：低频被显著激活
        if low >= self.th_kick:
            out.append(("kick", float(min(1.0, low))))

        # 军鼓：中频激活，且高频相对低频足够突出（排除底鼓的宽带瞬态）
        if (mid >= self.th_snare_mid and high >= self.th_snare_high
                and high >= self.th_snare_high_ratio * max(low, 1e-6)):
            out.append(("snare", float(min(1.0, 0.5 * (mid + high)))))

        # 镲类：超高频被激活，用「绝对残余 + 相对本曲常态」双重判据区分开/闭镲
        if vhigh >= self.th_hat:
            is_open = (sustain_abs >= self.th_open_hat_sustain_abs
                       and sustain_rel >= self.th_open_hat_sustain_rel)
            out.append(("open_hat" if is_open else "closed_hat",
                        float(min(1.0, vhigh))))
            # 镲片要求拖得更久，绝对与相对阈值同时提高
            if (vhigh >= 0.60 and sustain_abs >= self.th_crash_sustain_abs
                    and sustain_rel >= self.th_crash_sustain_rel):
                out.append(("crash", 0.70))

        # 通鼓：低频有激活但不如底鼓强，且低频衰减明显更长
        if (self.th_low_tom_min <= low < self.th_low_tom_max
                and 100.0 <= decay_low_ms <= 700.0):
            out.append(("low_tom", float(min(1.0, low + mid))))

        # 同一时刻最多保留 3 个鼓件，避免规则叠加出四五个音的虚假击打
        out.sort(key=lambda kv: -kv[1])
        return out[:3]

    # ---------- 主流程 ----------

    def transcribe(self, audio, sr: int | None = None,
                   min_confidence: float | None = None,
                   instrument: int = 0) -> StemNotes:
        """把（已分离的）鼓轨音频转录为鼓组音符。"""
        from ..ingest import resample

        if isinstance(audio, (str,)):
            import soundfile as sf

            data, file_sr = sf.read(str(audio), dtype="float32", always_2d=True)
            data = data.T
        else:
            data = np.asarray(audio, dtype=np.float32)
            if data.ndim == 1:
                data = data[None, :]
            file_sr = int(sr or self.sr)

        if file_sr != self.sr:
            data = resample(data, file_sr, self.sr)

        mono = self._to_mono(audio, sr)
        detections, detect_stats = self.detect(mono)

        threshold = 0.10 if min_confidence is None else float(min_confidence)
        notes: list[NoteEvent] = []
        hits: list[DrumHit] = []
        rejected = 0

        for d in detections:
            n = d["norm"]
            accepted_here = 0
            t = d["time"]
            vel = max(clamp_velocity(127.0 * min(1.0, d["rel_energy"])), 30)
            for kind, conf in d["candidates"]:
                if kind not in self.mapping or conf < threshold:
                    rejected += 1
                    continue
                pitch = int(self.mapping[kind])
                # 鼓音符给固定短时长：MIDI 打击乐靠 note-on 触发，
                # 时值过长会在部分音源上产生尾巴叠加。
                dur = 0.08 if kind in ("closed_hat", "kick", "snare") else 0.25
                # 同拍多鼓时略降次要鼓件的力度，避免听感浑浊
                v = vel if accepted_here == 0 else max(30, int(vel * 0.85))
                notes.append(NoteEvent(start=t, end=t + dur, pitch=pitch, velocity=v))
                hits.append(DrumHit(time=t, kind=kind, velocity=v, confidence=conf,
                                    norm_low=n["low"], norm_mid=n["mid"],
                                    norm_high=n["high"], norm_vhigh=n["vhigh"],
                                    sustain_vhigh=d["sustain_vhigh"],
                                    sustain_rel=d["sustain_rel"],
                                    decay_low_ms=d["decay_low_ms"]))
                accepted_here += 1
            if accepted_here == 0:
                rejected += 1

        result = StemNotes(stem="drums", notes=notes, instrument=instrument, is_drum_track=True)
        counts: dict[str, int] = {}
        for h in hits:
            counts[h.kind] = counts.get(h.kind, 0) + 1
        result.meta = {
            "engine": "drum_band_flux",
            "onset_sensitivity": self.sensitivity,
            "peaks_detected": detect_stats.get("peaks", 0),
            "hits_accepted": len(notes),
            "hits_rejected": rejected,
            "min_confidence": threshold,
            "kick_max_hz": self.kick_max_hz,
            "hat_min_hz": self.hat_min_hz,
            "counts_by_kind": counts,
            "note": "启发式频段分类，非学习模型；在干净分离的鼓轨上表现最佳",
        }
        return result

    # ---------- 检测（供 transcribe 与诊断共用） ----------

    def detect(self, mono: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """检测所有 onset，并给出每个 onset 的特征与候选鼓件。

        把这一步从 transcribe 里单独抽出来，是为了让诊断工具看到的数据
        与实际参与判定的数据完全一致。否则「调试时看到一套数、运行时用另一套数」
        会让排查变成猜谜。
        """
        mag = _magnitude_spectrogram(mono, N_FFT, HOP)
        if mag.shape[0] < 2:
            return [], {"frames": int(mag.shape[0]), "peaks": 0}

        flux = self._flux(mag)
        band_norm = self._band_norm(flux)
        energy = band_norm.sum(axis=1).astype(np.float32)
        peaks = self._pick_peaks(energy, self.sensitivity)

        # 力度参考：以 95 分位为满力度，避免个别尖峰把整首压暗
        ref = float(np.percentile(energy[peaks], 95)) if peaks.size else 1.0
        ref = max(ref, 1e-9)

        band_names = ("low", "mid", "high", "vhigh")
        band_masks = {"low": self.mask_low, "mid": self.mask_mid,
                      "high": self.mask_high, "vhigh": self.mask_vhigh}

        dets: list[dict[str, Any]] = []
        for p in peaks:
            raw = {k: float(flux[p, i]) for i, k in enumerate(band_names)}
            norm = {k: float(band_norm[p, i]) for i, k in enumerate(band_names)}
            dominant = max(norm, key=lambda k: norm[k])
            # 镲类用高频残余能量比，底鼓/通鼓用低频衰减时间。
            # 两个判据各自测在真正相关的频段上 —— 混用一个「主导频段」
            # 在同拍时必然测错对象。
            sustain_vhigh = self._sustain_ratio(mag, p, self.mask_vhigh)
            decay_low = self._decay_ms(mag, p, self.mask_low)
            # 时间补偿：帧 p 的窗口覆盖 [p*HOP, p*HOP+N_FFT)，其中心才是
            # 该帧能量真正代表的时刻。不补这半个窗长会系统性提前约 23ms。
            t = (p * HOP + N_FFT // 2) / self.sr
            dets.append({
                "frame": int(p),
                "time": float(t),
                "energy": float(energy[p]),
                "rel_energy": float(energy[p] / ref),
                "flux": raw,
                "norm": norm,
                "dominant_band": dominant,
                "sustain_vhigh": sustain_vhigh,
                "decay_low_ms": decay_low,
                "candidates": [],
            })

        # 残余能量比必须按本曲自身的典型水平归一，再判「是否异常拖长」。
        #
        # 这一步是必需的，不是锦上添花。真实音乐的高频内容通常是连续的
        # （踩镲密集、镲片与残响不断），击打后 70~163ms 的窗口里几乎总有
        # 别的高频事件，用绝对阈值会让所有镲都被判成开镲 ——
        # 实测某电子曲目 30 秒内输出 135 个开镲、67 个 crash，而闭镲只有 4 个，
        # 完全颠倒。合成测试没暴露这个问题，因为那里的击打彼此孤立。
        # 改为与中位数对比后，「明显比本曲常态更拖长」才是开镲或镲片。
        if dets:
            sustains = np.asarray([d["sustain_vhigh"] for d in dets], dtype=np.float64)
            baseline = float(np.median(sustains))
            baseline = max(baseline, 1e-6)
            for d in dets:
                d["sustain_rel"] = float(d["sustain_vhigh"] / baseline)
                d["candidates"] = self._classify_multi(
                    d["norm"], d["sustain_vhigh"], d["sustain_rel"], d["decay_low_ms"])
            self._last_sustain_baseline = baseline

        stats = {
            "frames": int(mag.shape[0]),
            "duration_sec": round(mag.shape[0] * HOP / self.sr, 3),
            "peaks": int(peaks.size),
            "sensitivity": self.sensitivity,
            "ref_energy": ref,
            "band_p95": {k: float(np.percentile(flux[:, i], 95))
                         for i, k in enumerate(band_names)},
            "thresholds": {
                "th_kick": self.th_kick,
                "th_snare_mid": self.th_snare_mid,
                "th_snare_high": self.th_snare_high,
                "th_hat": self.th_hat,
            },
        }
        return dets, stats

    def _to_mono(self, audio, sr: int | None) -> np.ndarray:
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
        return data.mean(axis=0) if data.shape[0] > 1 else data.reshape(-1)

    def analyze(self, audio, sr: int | None = None,
                limit: int = 60) -> dict[str, Any]:
        """诊断接口：返回中间量与每个 onset 的完整特征。

        调参或排查「为什么某类鼓件检不出来」时用它，比反复听结果有效得多。
        """
        mono = self._to_mono(audio, sr)
        dets, stats = self.detect(mono)
        stats["classifier_rules"] = {
            "kick": f"norm_low>={self.th_kick}",
            "snare": f"norm_mid>={self.th_snare_mid} 且 norm_high>={self.th_snare_high} "
                     f"且 norm_high>={self.th_snare_high_ratio}*norm_low",
            "hat": f"norm_vhigh>={self.th_hat}；开镲需 sustain_abs>="
                   f"{self.th_open_hat_sustain_abs} 且 sustain_rel>="
                   f"{self.th_open_hat_sustain_rel}（相对本曲中位数）",
            "crash": f"norm_vhigh>=0.60 且 sustain_abs>={self.th_crash_sustain_abs} "
                     f"且 sustain_rel>={self.th_crash_sustain_rel}",
            "low_tom": f"{self.th_low_tom_min}<=norm_low<{self.th_low_tom_max} "
                       f"且 100<=低频段 decay<=700ms",
        }
        stats["sustain_baseline"] = round(self._last_sustain_baseline, 6)
        stats["bands_hz"] = {"low": BAND_LOW, "mid": BAND_MID,
                             "high": BAND_HIGH, "vhigh": BAND_VHIGH}
        return {"stats": stats,
                "detections": [
                    {**d,
                     "norm": {k: round(v, 3) for k, v in d["norm"].items()},
                     "flux": {k: round(v, 5) for k, v in d["flux"].items()},
                     "sustain_vhigh": round(d["sustain_vhigh"], 4),
                     "sustain_rel": round(d.get("sustain_rel", 0.0), 3),
                     "decay_low_ms": round(d["decay_low_ms"], 1),
                     "candidates": [(k, round(c, 3)) for k, c in d["candidates"]]}
                    for d in dets[:limit]
                ]}
