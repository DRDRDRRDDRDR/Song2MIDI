"""Basic Pitch ONNX 音高转录引擎。

为何自己写推理封装而不直接调用 basic-pitch 的 predict()：
    1. 官方包在 Windows + Python>=3.11 下被元数据强制要求 tensorflow<2.15.1，
       而 TF 2.15 不支持 3.12+，pip 解析阶段即失败。实际算法并不需要 TF。
    2. 官方 run_inference 逐窗口 batch=1 推理，本机 CPU 上偏慢；
       ONNX 输入 batch 维是动态的，可以一次推多个窗口，实测明显更快。
    3. 四轨方案需要按分轨限制复音数（贝斯限 1 音、人声限 2 音），
       官方接口没有这个参数，必须在音符层面加过滤。

算法本身不做任何改动：窗口切分、重叠裁剪、音符提取全部沿用官方实现，
参数与官方 inference.py 逐项对齐（见下方常量）。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..config import Config
from ..notes import NoteEvent, StemNotes, clamp_velocity
from ._vendor import note_creation as bp_nc

__all__ = ["PitchTranscriber", "ONNX_MODEL_NAME"]

ONNX_MODEL_NAME = "nmp.onnx"

# ---- 与官方 basic_pitch/constants.py 完全一致的参数，改动会破坏模型对齐 ----
FFT_HOP = 256
AUDIO_SAMPLE_RATE = 22050
AUDIO_WINDOW_LENGTH = 2
AUDIO_N_SAMPLES = AUDIO_SAMPLE_RATE * AUDIO_WINDOW_LENGTH - FFT_HOP  # 43844
ANNOTATIONS_FPS = AUDIO_SAMPLE_RATE // FFT_HOP                       # 86
MIDI_OFFSET = 21

# 官方 run_inference 使用 30 帧重叠
N_OVERLAP_FRAMES = 30
OVERLAP_LEN = N_OVERLAP_FRAMES * FFT_HOP              # 7680
HOP_SIZE = AUDIO_N_SAMPLES - OVERLAP_LEN              # 36164

# ONNX 张量名，取自官方 Model.predict 的 ONNX 分支
INPUT_NAME = "serving_default_input_2:0"
OUTPUT_NAMES = {
    "note": "StatefulPartitionedCall:1",
    "onset": "StatefulPartitionedCall:2",
    "contour": "StatefulPartitionedCall:0",
}


class TranscriptionError(RuntimeError):
    pass


class PitchTranscriber:
    """Basic Pitch 多音转录。单例复用即可，ONNX session 创建有一定开销。"""

    def __init__(self, cfg: Config, model_path: str | Path | None = None,
                 providers: list[str] | None = None,
                 batch_size: int = 8) -> None:
        self.cfg = cfg
        self.model_path = Path(model_path) if model_path else self._default_model_path()
        if not self.model_path.is_file():
            raise TranscriptionError(
                f"未找到 ONNX 模型: {self.model_path}\n"
                f"请确认 models/{ONNX_MODEL_NAME} 存在。"
            )
        self.batch_size = max(1, int(batch_size))

        import onnxruntime as ort

        avail = ort.get_available_providers()
        if providers:
            use = [p for p in providers if p in avail] or ["CPUExecutionProvider"]
        else:
            # 该模型是小型 CNN（230KB），CPU 上已足够快；
            # 交给 GPU 反而要付传输开销，故默认 CPU。
            use = ["CPUExecutionProvider"]

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # 限制线程数，避免多个 worker 进程互相抢占导致整体变慢
        opts.intra_op_num_threads = int(cfg.get("transcription.onnx_threads", 0)) or 0
        self.session = ort.InferenceSession(str(self.model_path), sess_options=opts, providers=use)
        self.providers = self.session.get_providers()
        self._input = self.session.get_inputs()[0].name

    def _default_model_path(self) -> Path:
        models_dir = self.cfg.path("paths.models_dir")
        return models_dir / ONNX_MODEL_NAME

    # ---------- 音频准备 ----------

    def _prepare_audio(self, audio, sr: int | None = None) -> tuple[np.ndarray, int]:
        """统一为 22050Hz 单声道 float32 一维数组。"""
        if isinstance(audio, (str, Path)):
            import soundfile as sf

            data, file_sr = sf.read(str(audio), dtype="float32", always_2d=True)
            data = data.T
            if sr is not None and sr != file_sr:
                from ..ingest import resample

                data = resample(data, file_sr, sr)
                file_sr = sr
        else:
            data = np.asarray(audio, dtype=np.float32)
            file_sr = int(sr or AUDIO_SAMPLE_RATE)
            if data.ndim == 1:
                data = data[None, :]

        if file_sr != AUDIO_SAMPLE_RATE:
            from ..ingest import resample

            data = resample(data, file_sr, AUDIO_SAMPLE_RATE)
            file_sr = AUDIO_SAMPLE_RATE

        mono = data.mean(axis=0) if data.ndim == 2 and data.shape[0] > 1 else data.reshape(-1)
        return np.ascontiguousarray(mono, dtype=np.float32), AUDIO_SAMPLE_RATE

    # ---------- 推理 ----------

    def _window(self, mono: np.ndarray) -> np.ndarray:
        """按官方方式切窗：前置半个重叠长度的零，末尾补零到整窗。"""
        original_length = mono.shape[0]
        padded = np.concatenate([
            np.zeros(int(OVERLAP_LEN / 2), dtype=np.float32), mono
        ])
        windows: list[np.ndarray] = []
        for i in range(0, padded.shape[0], HOP_SIZE):
            w = padded[i:i + AUDIO_N_SAMPLES]
            if w.shape[0] < AUDIO_N_SAMPLES:
                w = np.pad(w, (0, AUDIO_N_SAMPLES - w.shape[0]))
            windows.append(w)
        if not windows:
            windows.append(np.zeros(AUDIO_N_SAMPLES, dtype=np.float32))
        return np.stack(windows).astype(np.float32)[:, :, None], original_length

    def _unwrap(self, output: np.ndarray, original_length: int) -> np.ndarray:
        """官方 unwrap_output 的等价实现：每窗掐掉一半重叠帧后拼接，再裁到原始长度。"""
        n_olap = int(0.5 * N_OVERLAP_FRAMES)
        if n_olap > 0:
            output = output[:, n_olap:-n_olap, :]
        shape = output.shape
        flat = output.reshape(shape[0] * shape[1], shape[2])
        n_frames = int(math.floor(original_length * (ANNOTATIONS_FPS / AUDIO_SAMPLE_RATE)))
        return flat[:n_frames, :]

    def raw_output(self, audio, sr: int | None = None) -> dict[str, np.ndarray]:
        """跑模型，返回与官方 run_inference 等价的 note/onset/contour 三矩阵。"""
        mono, _ = self._prepare_audio(audio, sr)
        batch, original_length = self._window(mono)

        collected: dict[str, list[np.ndarray]] = {"note": [], "onset": [], "contour": []}
        for start in range(0, batch.shape[0], self.batch_size):
            chunk = batch[start:start + self.batch_size]
            outs = self.session.run(list(OUTPUT_NAMES.values()), {self._input: chunk})
            for key, arr in zip(OUTPUT_NAMES.keys(), outs):
                collected[key].append(arr)

        return {
            key: self._unwrap(np.concatenate(vals, axis=0), original_length)
            for key, vals in collected.items()
        }

    # ---------- 音符提取 ----------

    def transcribe(
        self,
        audio,
        sr: int | None = None,
        stem: str = "mix",
        onset_threshold: float | None = None,
        frame_threshold: float | None = None,
        min_note_ms: float | None = None,
        min_freq: float | None = None,
        max_freq: float | None = None,
        max_polyphony: int | None = None,
        melodia_trick: bool | None = None,
        include_pitch_bends: bool = False,
        instrument: int = 1,
        extra_meta: dict[str, Any] | None = None,
    ) -> StemNotes:
        """把音频转录为音符。参数留空则取 config.yaml 的值。"""
        onset_threshold = self._pick(onset_threshold, "transcription.onset_threshold", 0.5)
        frame_threshold = self._pick(frame_threshold, "transcription.frame_threshold", 0.3)
        min_note_ms = self._pick(min_note_ms, "transcription.minimum_note_length_ms", 58.0)
        if max_polyphony is None:
            max_polyphony = int(self.cfg.get(f"transcription.per_stem.{stem}.max_polyphony", 0)) or 0
        if melodia_trick is None:
            melodia_trick = bool(self.cfg.get("transcription.melodia_trick", True))

        # 官方接口以「帧」为单位，换算时用官方 ANNOTATIONS_FPS 保证与模型时序一致
        min_note_len = max(1, int(round(min_note_ms / 1000.0 * ANNOTATIONS_FPS)))

        output = self.raw_output(audio, sr)
        midi, note_events = bp_nc.model_output_to_notes(
            output,
            onset_thresh=float(onset_threshold),
            frame_thresh=float(frame_threshold),
            infer_onsets=True,
            min_note_len=min_note_len,
            min_freq=min_freq,
            max_freq=max_freq,
            include_pitch_bends=bool(include_pitch_bends),
            multiple_pitch_bends=False,
            melodia_trick=bool(melodia_trick),
        )

        notes: list[NoteEvent] = []
        for ev in note_events:
            start, end, pitch, amplitude = ev[0], ev[1], ev[2], ev[3]
            bends = list(ev[4]) if len(ev) > 4 and ev[4] else None
            notes.append(NoteEvent(
                start=float(start), end=float(end), pitch=int(pitch),
                velocity=clamp_velocity(127 * float(amplitude)),
                pitch_bends=bends if include_pitch_bends else None,
            ))

        dropped = 0
        if max_polyphony and max_polyphony > 0:
            notes, dropped = limit_polyphony(notes, max_polyphony)

        result = StemNotes(stem=stem, notes=notes, instrument=instrument)
        result.meta = {
            "engine": "basic_pitch_onnx",
            "model": self.model_path.name,
            "providers": self.providers,
            "onset_threshold": onset_threshold,
            "frame_threshold": frame_threshold,
            "min_note_len_frames": min_note_len,
            "min_note_ms": min_note_ms,
            "min_freq": min_freq,
            "max_freq": max_freq,
            "max_polyphony": max_polyphony,
            "melodia_trick": melodia_trick,
            "include_pitch_bends": include_pitch_bends,
            "notes_before_polyphony_limit": len(notes) + dropped,
            "notes_dropped_by_polyphony": dropped,
            "output_frames": int(output["note"].shape[0]),
        }
        if extra_meta:
            result.meta.update(extra_meta)
        return result

    def _pick(self, value, dotted: str, default):
        if value is not None:
            return value
        got = self.cfg.get(dotted, default)
        return default if got is None else got


def limit_polyphony(notes: Iterable[NoteEvent], max_polyphony: int
                    ) -> tuple[list[NoteEvent], int]:
    """按时间轴限制同时发声的音符数，超出者按较短音符优先剔除。

    为什么是「较短优先」：短音符更可能是转录残留的杂音，
    而持续的长音符通常对应真实的声部线条。贝斯轨限 1 音时这个策略效果尤其明显。
    """
    items = list(notes)
    if max_polyphony <= 0 or not items:
        return items, 0

    # 收集所有时间边界作为切片点
    bounds = sorted({n.start for n in items} | {n.end for n in items})
    removed: set[int] = set()

    for i in range(len(bounds) - 1):
        t0, t1 = bounds[i], bounds[i + 1]
        if t1 - t0 <= 1e-9:
            continue
        mid = (t0 + t1) / 2.0
        active = [idx for idx, n in enumerate(items)
                  if idx not in removed and n.start <= mid < n.end]
        if len(active) <= max_polyphony:
            continue
        # 长音符优先保留，短音符先淘汰；等长时保留力度更大的
        active.sort(key=lambda idx: (-items[idx].duration, -items[idx].velocity))
        for idx in active[max_polyphony:]:
            removed.add(idx)

    kept = [n for idx, n in enumerate(items) if idx not in removed]
    return kept, len(removed)
