"""端到端编排：一首歌 → 多轨 MIDI。

各阶段的职责边界：
    摄取    任意格式 → 标准 wav（统一格式差异）
    分离    wav → vocals / drums / bass / other 四轨
    转录    各轨用各自最合适的算法（音高模型 / 频段分类 / 模板匹配）
    后处理  量化、合并、去重、归一（决定 MIDI 是否「能用」）
    导出    多轨 MIDI + 音符 JSON

模型对象全部惰性创建并在进程内复用。Demucs 权重加载一次需要数秒，
ONNX 会话创建也要数百毫秒，逐文件重建会让批量处理的耗时被加载开销主导。
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import Config, load_config
from .export import MidiExporter
from .ingest import Ingest, IngestError
from .notes import StemNotes
from .pool import JobResult, StageLimiter
from .postprocess import PostProcessor, TempoInfo, estimate_tempo

__all__ = ["Song2MidiPipeline", "PipelineOptions", "PIPELINE_VERSION"]

PIPELINE_VERSION = "0.1.0"

# 各分轨的转录方式
STEM_ENGINE = {
    "vocals": "pitch",
    "bass": "pitch",
    "other": "pitch",
    "drums": "drums",
    "mix": "pitch",
}


@dataclass
class PipelineOptions:
    # None = 自动检测（无鼓+无贝斯的独奏曲自动跳过分离）；
    # True = 强制四轨分离；False = 强制整首直转
    separate: bool | None = None
    transcribe_drums: bool = True
    transcribe_chords: bool = True
    stems: tuple[str, ...] = ("vocals", "drums", "bass", "other")
    start: float | None = None
    duration: float | None = None
    chord_source: str = "other"      # other | harmonic | mix
    keep_intermediate: bool | None = None
    quantize: bool | None = None


class PipelineError(RuntimeError):
    pass


class Song2MidiPipeline:
    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg or load_config()
        self.ingest = Ingest(self.cfg)
        self.post = PostProcessor(self.cfg)
        self.exporter = MidiExporter(self.cfg)
        self._sep = None
        self._tr = None
        self._drum = None
        self._chord = None
        self._piano = None
        # 本次任务的独奏检测结果，供 _transcribe_stem 决定是否用钢琴引擎
        self._solo_info: dict | None = None

    # ---------- 惰性模型 ----------

    @property
    def separator(self):
        if self._sep is None:
            # 经由工厂创建，具体是 Demucs 还是 BS-RoFormer 由 config 决定。
            # 两者对外接口一致，管线无需关心。
            from .engines import create_separator

            self._sep = create_separator(self.cfg)
        return self._sep

    @property
    def transcriber(self):
        if self._tr is None:
            from .engines.pitch_onnx import PitchTranscriber

            self._tr = PitchTranscriber(self.cfg)
        return self._tr

    @property
    def piano_engine(self):
        """钢琴专用转录引擎（ByteDance 高分辨率钢琴转录）。

        惰性创建：只有真正用到时才 import 那些依赖、才加载 164MB 权重。
        """
        if self._piano is None:
            from .engines.pitch_piano import PianoTranscriber

            self._piano = PianoTranscriber(self.cfg)
        return self._piano

    @property
    def drum_engine(self):
        if self._drum is None:
            from .engines.drums import DrumTranscriber

            self._drum = DrumTranscriber(self.cfg)
        return self._drum

    @property
    def chord_engine(self):
        if self._chord is None:
            from .engines.chords import ChordRecognizer

            self._chord = ChordRecognizer(self.cfg)
        return self._chord

    # ---------- 主流程 ----------

    def process(self, src: str | Path, opts: PipelineOptions | None = None,
                limiters: tuple[StageLimiter, StageLimiter] | None = None,
                on_event=None) -> tuple[list[StemNotes], dict[str, Any]]:
        """处理单个文件，返回 (分轨结果, 汇总信息)。"""
        opts = opts or PipelineOptions()
        sep_limiter, tr_limiter = limiters or (None, None)

        def emit(event: str, **data: Any) -> None:
            if on_event:
                try:
                    on_event(event, data)
                except Exception:
                    pass

        timings: dict[str, float] = {}
        warnings: list[str] = []

        # ---- 1. 摄取 ----
        t = time.perf_counter()
        emit("stage", stage="ingest")
        info = self.ingest.prepare(src, start=opts.start, duration=opts.duration)
        timings["ingest"] = time.perf_counter() - t
        warnings.extend(info.warnings or [])
        emit("ingest_done", info=info.to_dict())

        basename = Path(info.source_path).stem
        audio, sr = self.ingest.load_array(info.wav_path, mono=False)

        # ---- 音频预处理（可选）----
        # 只做高通。实测 noisereduce 之类的谱门控降噪会改动 58% 的信号却无收益，
        # 还把钢琴低音区削掉（音域 28–88 → 38–88），因此不提供。
        hp = float(self.cfg.get("preprocess.highpass_hz", 0) or 0)
        if hp > 0:
            from .preprocess import apply_highpass

            t = time.perf_counter()
            audio, hp_info = apply_highpass(audio, sr, hp)
            timings["preprocess"] = time.perf_counter() - t
            emit("preprocess_done", **hp_info)
            if hp_info.get("applied"):
                warnings.append(
                    f"已施加 {hp:.0f}Hz 高通（改动信号 {hp_info.get('changed_pct')}%）")
            elif hp_info.get("skipped"):
                warnings.append(f"高通未生效：{hp_info['skipped']}")

        # ---- 2. 分离（含自动检测）----
        stems_audio: dict[str, np.ndarray] = {}
        do_separate = opts.separate
        solo_detect = None
        if do_separate is None:
            # 自动检测：无鼓+无贝斯的独奏曲（钢琴/吉他/弦乐）强行四轨分离
            # 会产生大量假鼓并割裂旋律，因此跳过分离、整首直转。
            from .solo_detect import detect_solo

            t = time.perf_counter()
            solo_detect = detect_solo(
                audio, sr,
                high_ratio_limit=float(self.cfg.get("solo_detect.high_ratio_limit", 0.02)),
                low_ratio_limit=float(self.cfg.get("solo_detect.low_ratio_limit", 0.03)),
                analysis_sr=int(self.cfg.get("solo_detect.analysis_sr", 22050)),
            )
            timings["solo_detect"] = time.perf_counter() - t
            # 记到实例上：_transcribe_stem 靠它判断要不要改用钢琴专用引擎
            self._solo_info = solo_detect
            do_separate = not solo_detect["solo"]
            emit("solo_detect_done", **solo_detect)
            if solo_detect["solo"]:
                warnings.append(
                    f"自动检测：{solo_detect['reason']} —— 已跳过分离，整首直转")

        if do_separate:
            emit("stage", stage="separate")
            t = time.perf_counter()
            ctx = sep_limiter if sep_limiter is not None else _NullCtx()
            with ctx:
                sep = self.separator
                stems_audio = sep.separate(audio, sr)
            timings["separate"] = time.perf_counter() - t
            # 分轨音频落盘。分离是整个流程里最贵的一步，把中间结果留下来
            # 有两个实际价值：一是前端可以逐轨试听，从而分辨「是分离错了
            # 还是转录错了」—— 这是排查音质问题时唯一有效的办法；
            # 二是分离耗时远大于转录，留在盘上就不必为改参数反复重算。
            if opts.keep_intermediate is not False and self.cfg.get("logging.keep_intermediate", True):
                stem_dir = self.cfg.path("paths.work_dir") / "stems" / basename
                self._write_stems(stems_audio, stem_dir, sr, warnings)
            emit("separate_done", stems=sorted(stems_audio.keys()),
                 seconds=round(timings["separate"], 2),
                 device=self.separator.device)
        else:
            stems_audio = {"mix": audio}
            timings["separate"] = 0.0
            emit("separate_skipped")

        # ---- 3. 节拍估计（用鼓轨最准，没有则用原始混音）----
        t = time.perf_counter()
        beat_src = stems_audio.get("drums")
        if beat_src is None:
            beat_src = stems_audio.get("mix", audio)
        beat_mono = (beat_src.mean(axis=0) if beat_src.ndim == 2 and beat_src.shape[0] > 1
                     else np.asarray(beat_src).reshape(-1))
        tempo: TempoInfo = self.post.resolve_tempo(beat_mono, sr)
        timings["tempo"] = time.perf_counter() - t
        emit("tempo_done", tempo=tempo.to_dict())

        # ---- 4. 分轨转录 ----
        results: list[StemNotes] = []
        want = set(opts.stems)
        if not do_separate:
            want = {"mix"}

        for stem in [s for s in ("vocals", "drums", "bass", "other", "mix") if s in want]:
            if stem not in stems_audio:
                warnings.append(f"分离结果缺少 {stem} 轨，已跳过")
                continue
            if stem == "drums" and not opts.transcribe_drums:
                continue

            emit("stage", stage=f"transcribe:{stem}")
            t = time.perf_counter()
            ctx = tr_limiter if tr_limiter is not None else _NullCtx()
            try:
                with ctx:
                    sn = self._transcribe_stem(stem, stems_audio[stem], sr)
            except Exception as e:
                warnings.append(f"{stem} 轨转录失败: {e.__class__.__name__}: {e}")
                emit("stem_failed", stem=stem, error=str(e))
                continue
            timings[f"transcribe_{stem}"] = time.perf_counter() - t
            emit("stem_done", stem=stem, notes=len(sn), seconds=round(timings[f"transcribe_{stem}"], 2),
                 stats=sn.stats())
            results.append(sn)

        # ---- 5. 和弦 ----
        if opts.transcribe_chords and "chords" not in want:
            chord_audio = self._chord_source(stems_audio, opts.chord_source)
            if chord_audio is not None:
                emit("stage", stage="transcribe:chords")
                t = time.perf_counter()
                try:
                    with (tr_limiter if tr_limiter is not None else _NullCtx()):
                        ch = self.chord_engine.transcribe(chord_audio, sr)
                    if ch.notes:
                        timings["transcribe_chords"] = time.perf_counter() - t
                        emit("stem_done", stem="chords", notes=len(ch),
                             seconds=round(timings["transcribe_chords"], 2), stats=ch.stats())
                        results.append(ch)
                except Exception as e:
                    warnings.append(f"和弦识别失败: {e.__class__.__name__}: {e}")

        # ---- 6. 后处理 ----
        emit("stage", stage="postprocess")
        t = time.perf_counter()

        # 节拍置信度过低时放弃量化。
        # 实测：钢琴曲只取 20 秒片段时 BPM 会被估成 80.6（实际 123），
        # 置信度仅 0.36；而 ≥40 秒就稳定在 122.86（置信度 0.46+）。
        # 网格错了还硬量化，等于把所有音符搬到错误位置 —— 不如不量化。
        quantize_opt = opts.quantize
        conf = float(getattr(tempo, "confidence", 1.0) or 0.0)
        min_conf = float(self.cfg.get("postprocess.min_tempo_confidence", 0.40))
        if quantize_opt is not False and conf < min_conf:
            warnings.append(
                f"节拍置信度仅 {conf:.2f}（低于 {min_conf:.2f}），已跳过量化以避免"
                f"把音符对齐到错误的网格。可增大处理时长，或在配置里手动指定 BPM。")
            quantize_opt = False

        processed: list[StemNotes] = []
        for sn in results:
            try:
                processed.append(self.post.process(sn, tempo, quantize=quantize_opt))
            except Exception as e:
                warnings.append(f"{sn.stem} 后处理失败，使用原始音符: {e}")
                processed.append(sn)
        timings["postprocess"] = time.perf_counter() - t

        summary = {
            "source": info.source_path,
            "basename": basename,
            "duration_sec": round(info.duration_sec, 2),
            "sample_rate": sr,
            "containers": {"codec": info.source_codec, "container": info.source_container,
                           "bits": info.source_bits},
            "separated": do_separate,
            "solo_detect": solo_detect,
            "separator": (
                {"engine": "roformer" if "RoFormer" in type(self.separator).__name__ else "demucs",
                 "model": getattr(self.separator, "model_name", None),
                 "device": getattr(self.separator, "device", None)}
                if do_separate else None
            ),
            "tempo": tempo.to_dict(),
            "timings": {k: round(v, 2) for k, v in timings.items()},
            "stems": [s.stats() for s in processed],
            "warnings": warnings,
            "pipeline_version": PIPELINE_VERSION,
        }
        return processed, summary

    def _write_stems(self, stems_audio: dict[str, np.ndarray], dst_dir: Path,
                     sr: int, warnings: list[str]) -> dict[str, str]:
        """把分离出的分轨写成 16bit wav，供前端试听与人工核对。"""
        import soundfile as sf

        paths: dict[str, str] = {}
        try:
            dst_dir.mkdir(parents=True, exist_ok=True)
            for name, arr in stems_audio.items():
                dst = dst_dir / f"{name}.wav"
                data = np.asarray(arr, dtype=np.float32)
                # soundfile 要 (samples, channels)，本项目内部统一 (channels, samples)
                if data.ndim == 2 and data.shape[0] < data.shape[1]:
                    data = data.T
                sf.write(str(dst), np.clip(data, -1.0, 1.0), sr, subtype="PCM_16")
                paths[name] = str(dst)
        except Exception as e:
            # 落盘失败不应中断主流程 —— 它只是附带的中间产物
            warnings.append(f"分轨音频落盘失败（不影响 MIDI 输出）: {e}")
        return paths

    def _piano_selected(self, stem: str) -> bool:
        """是否改用钢琴专用引擎。

        transcription.engine: auto | basic_pitch | piano
          - piano       强制用
          - basic_pitch 强制不用
          - auto        仅在「本轨是整首直转的 mix」且「检测为无鼓无贝斯的
                        独奏」时启用 —— 钢琴独奏是这类曲目的绝大多数情形；
                        用户仍可在界面把引擎改成 basic_pitch 覆盖。
        """
        mode = str(self.cfg.get("transcription.engine", "auto") or "auto").lower()
        if mode == "piano":
            return True
        if mode == "basic_pitch":
            return False
        # auto
        if stem != "mix":
            return False
        return bool((self._solo_info or {}).get("solo"))

    def _transcribe_stem(self, stem: str, audio: np.ndarray, sr: int) -> StemNotes:
        engine = STEM_ENGINE.get(stem, "pitch")
        if engine == "drums":
            return self.drum_engine.transcribe(audio, sr)

        if self._piano_selected(stem):
            try:
                return self.piano_engine.transcribe(audio, sr, stem=stem, instrument=0)
            except Exception as e:
                # 钢琴引擎不可用（未装包 / 权重缺失）时回退到通用引擎，
                # 不能让整首任务因为一个可选增强而失败。
                import warnings

                warnings.warn(f"钢琴专用引擎不可用，回退到通用引擎：{e}")
                self._piano_fallback_reason = f"{e.__class__.__name__}: {e}"

        stem_cfg = self.cfg.section("transcription").get("per_stem", {}).get(stem, {})
        return self.transcriber.transcribe(
            audio, sr=sr, stem=stem,
            min_freq=stem_cfg.get("fmin"),
            max_freq=stem_cfg.get("fmax"),
            max_polyphony=stem_cfg.get("max_polyphony"),
        )

    def _chord_source(self, stems_audio: dict[str, np.ndarray], mode: str):
        """选择和弦识别用哪条音频。

        用「贝斯 + other」叠加而不是只用 other：和弦的根音往往落在贝斯上，
        单看 other 容易把转位和弦判成原位。叠加后根音信息更完整。
        """
        if mode == "mix" and "mix" in stems_audio:
            return stems_audio["mix"]
        if mode == "other":
            # 未做分离时没有 other 轨，回退到整首混音，否则和弦识别会被静默跳过。
            # 注意：这里必须用显式的 in 判断，不能写 `a or b` ——
            # numpy 数组的布尔求值会抛 "truth value of an array is ambiguous"，
            # 而且是在分离算了整整一轮之后才崩，代价很高。
            if "other" in stems_audio:
                return stems_audio["other"]
            return stems_audio.get("mix")
        # harmonic：贝斯降权后与 other 混合
        parts = []
        if "bass" in stems_audio:
            parts.append(("bass", 0.7))
        if "other" in stems_audio:
            parts.append(("other", 1.0))
        if not parts:
            # 同理，不能用 `a or b`：numpy 数组的布尔求值会抛异常
            if "other" in stems_audio:
                return stems_audio["other"]
            return stems_audio.get("mix")
        base = stems_audio[parts[0][0]]
        n = base.shape[1]
        acc = np.zeros_like(base)
        for name, gain in parts:
            arr = stems_audio[name]
            m = min(n, arr.shape[1])
            acc[:, :m] += arr[:, :m] * gain
        peak = float(np.abs(acc).max())
        if peak > 1.0:
            acc /= peak
        return acc

    # ---------- 落盘 ----------

    def export(self, stems: list[StemNotes], summary: dict[str, Any]) -> dict[str, Any]:
        bpm = summary.get("tempo", {}).get("bpm", 120.0)
        res = self.exporter.export(stems, summary["basename"], bpm=bpm,
                                   extra_meta=summary)
        payload = res.to_dict()
        # 写文件成功不等于文件正确，回读校验一次
        verifications = []
        for m in res.midi_files:
            v = self.exporter.verify(m)
            verifications.append(v)
            if not v.get("ok") or not v.get("total_notes"):
                res.warnings.append(f"MIDI 回读校验异常: {m} → {v}")
        payload["verification"] = verifications
        payload["warnings"] = res.warnings
        return payload

    # ---------- 供并行池调用的单文件任务 ----------

    def make_task(self, src: str | Path, opts: PipelineOptions | None = None):
        """生成 ParallelRunner 需要的可调用对象。"""
        opts = opts or PipelineOptions()

        def task(job_id: str, limiters) -> JobResult:
            t0 = time.perf_counter()
            res = JobResult(job_id=job_id, source=str(src))
            try:
                stems, summary = self.process(src, opts, limiters=limiters)
                res.duration_sec = summary.get("duration_sec", 0.0)
                export_info = self.export(stems, summary)
                res.stage_timings = summary.get("timings", {})
                res.payload = {"summary": summary, "export": export_info}
                res.ok = True
            except (IngestError, PipelineError) as e:
                res.error = f"{e.__class__.__name__}: {e}"
            except Exception as e:
                res.error = f"{e.__class__.__name__}: {e}\n{traceback.format_exc()[-1500:]}"
            res.elapsed_sec = time.perf_counter() - t0
            return res

        return task


class _NullCtx:
    """无限流器时使用的空上下文，让调用点无需分支判断。"""

    def __enter__(self):
        return self

    def __exit__(self, *exc: object) -> None:
        return None
