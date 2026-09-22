"""钢琴专用转录引擎（ByteDance 高分辨率钢琴转录）。

为什么需要它：
    通用引擎 Basic Pitch 的目标是「任意乐器的多音转录」，对钢琴曲存在两个
    可测量的短板 —— 实测同一段钢琴曲，它把大量音符**过早切断**
    （时值中位 366ms，而真实演奏约 699ms），且完全不输出踏板信息。

    本引擎基于 ByteDance 的 CRNN 高分辨率钢琴转录（MAESTRO 上
    note F1=0.9677 / pedal F1=0.9186），专为钢琴训练，
    输出 onset / offset / pitch / velocity **以及延音踏板**。

实测对比（同一 30 秒钢琴片段）：
    指标             Basic Pitch    本引擎
    碎片率(<60ms)      4.1%         1.3%
    时值中位           366 ms       699 ms
    音域              33–86        28–88
    推理速度           4.89x        9.37x
    踏板事件           无            36

对外接口与 PitchTranscriber 完全一致，管线无需区分。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from ..config import Config
from ..notes import NoteEvent, PedalEvent, StemNotes, clamp_velocity

__all__ = ["PianoTranscriber", "PianoTranscriberError"]

# 模型固定工作采样率（由训练配置决定，不能改）
MODEL_SR = 16000


class PianoTranscriberError(RuntimeError):
    pass


class PianoTranscriber:
    """ByteDance 钢琴转录推理封装。"""

    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg or Config()
        self._model = None
        self._device = None

    # ---------------- 模型 ----------------

    @property
    def device(self) -> str:
        if self._device is None:
            want = str(self.cfg.get("separator.device", "auto") or "auto").lower()
            if want == "cpu":
                self._device = "cpu"
            else:
                try:
                    import torch

                    if torch.cuda.is_available():
                        self._device = "cuda"
                    else:
                        self._device = "cpu"
                except Exception:
                    self._device = "cpu"
        return self._device

    def checkpoint_path(self) -> str:
        """按优先级定位模型权重。

        1. config 显式指定（transcription.piano.checkpoint）
        2. 项目 models/piano_transcription/ 下（随包分发，离线可用）
        3. 库的默认位置 ~/piano_transcription_inference_data/
        """
        explicit = self.cfg.get("transcription.piano.checkpoint")
        if explicit and Path(explicit).is_file():
            return str(explicit)

        name = "note_F1=0.9677_pedal_F1=0.9186.pth"
        for cand in (
            self.cfg.path("paths.models_dir") / "piano_transcription" / name,
            Path(os.path.expanduser("~")) / "piano_transcription_inference_data" / name,
        ):
            if Path(cand).is_file():
                return str(cand)
        # 都没找到就交给库自己下载（会用到 wget，Windows 上会失败，
        # 因此上层应在此之前提示用户）
        return str(Path(os.path.expanduser("~")) /
                   "piano_transcription_inference_data" / name)

    @property
    def model(self):
        if self._model is None:
            try:
                from piano_transcription_inference import PianoTranscription
            except ImportError as e:
                raise PianoTranscriberError(
                    "未安装 piano_transcription_inference。安装："
                    "pip install piano-transcription-inference audioread"
                ) from e

            ckpt = self.checkpoint_path()
            if not Path(ckpt).is_file():
                raise PianoTranscriberError(
                    f"找不到钢琴模型权重：{ckpt}\n"
                    "该库用 wget 下载模型，Windows 上没有 wget 会静默失败。"
                    "请手动下载后放到上述路径（可从 HuggingFace 镜像获取）。"
                )
            self._model = PianoTranscription(device=self.device,
                                            checkpoint_path=ckpt)
        return self._model

    # ---------------- 推理 ----------------

    def _to_model_input(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """转成模型要的 mono 1D @ 16000 Hz。

        模型内部会自行 padding 到 10 秒的整数倍，这里只需保证采样率与声道数。
        """
        y = np.asarray(audio, dtype=np.float32)
        if y.ndim == 2:
            y = y.mean(axis=0) if y.shape[0] > 1 else y[0]
        y = np.ascontiguousarray(y.reshape(-1))

        if sr != MODEL_SR:
            import librosa

            y = librosa.resample(y, orig_sr=sr, target_sr=MODEL_SR)
        return y

    def transcribe(self, audio: np.ndarray, sr: int, stem: str = "mix",
                   instrument: int = 0, **kwargs: Any) -> StemNotes:
        """转录为音符（含踏板）。

        接口与 PitchTranscriber.transcribe 对齐，多余的 kwargs 被忽略 ——
        钢琴模型不需要 per_stem 的复音/频域限制（它自带针对钢琴的解码策略）。
        """
        y = self._to_model_input(audio, sr)
        if y.size == 0:
            return StemNotes(stem=stem, instrument=instrument,
                             meta={"engine": "piano_transcription", "empty": True})

        # midi_path 传空字符串：该实现用 `if midi_path:` 判断是否落盘，
        # 传空即只算事件、不写文件，避免多余的磁盘 IO 与临时文件清理。
        result = self.model.transcribe(y, "")
        note_events = result.get("est_note_events") or []
        pedal_events = result.get("est_pedal_events") or []

        notes: list[NoteEvent] = []
        for ev in note_events:
            try:
                s = float(ev["onset_time"])
                e = float(ev["offset_time"])
                p = int(ev["midi_note"])
            except (KeyError, TypeError, ValueError):
                continue
            if e <= s:
                # 极少数情况下 offset 不晚于 onset，给一个最小可听时值，
                # 否则写出 MIDI 会得到零长度音符
                e = s + 0.03
            notes.append(NoteEvent(start=s, end=e, pitch=p,
                                   velocity=clamp_velocity(ev.get("velocity", 100))))

        pedals: list[PedalEvent] = []
        for ev in pedal_events:
            try:
                s = float(ev["onset_time"])
                e = float(ev["offset_time"])
            except (KeyError, TypeError, ValueError):
                continue
            if e > s:
                pedals.append(PedalEvent(start=s, end=e))

        sn = StemNotes(stem=stem, notes=notes, pedals=pedals, instrument=instrument)
        sn.meta = {
            "engine": "piano_transcription",
            "model": "CRNN note_F1=0.9677 pedal_F1=0.9186",
            "device": self.device,
            "model_sr": MODEL_SR,
            "input_sr": int(sr),
            "checkpoint": Path(self.checkpoint_path()).name,
            "notes_raw": len(note_events),
            "pedals": len(pedals),
        }
        return sn
