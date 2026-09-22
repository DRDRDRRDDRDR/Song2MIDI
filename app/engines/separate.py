"""Demucs 四轨音源分离。

输出 4 条分轨：vocals / drums / bass / other，对应 htdemucs 的 sources。

设计取舍：
    不使用 demucs 自带的音频文件读写路径（demucs.audio / torchaudio 解码），
    而是自己把 numpy 数组构造成张量后调用 apply_model。
    理由有两条：
      1. torchaudio 在 2.9+ 大幅重构了解码后端 API，而 demucs 4.1.0 发布于
         该重构之前，直接走文件路径有较大概率踩到接口不匹配。
      2. 上游的 ffmpeg 解码层已经产出统一 wav，再让 demucs 解码一次纯属重复，
         而且会引入第二套解码行为差异。
    正规化（减均值除标准差）与分块策略严格按官方 separate.py 实现，不做改动。

权重下载：htdemucs 的权重托管在 dl.fbaipublicfiles.com（已实测可达），
通过 TORCH_HOME 定向到项目 models/ 目录，不与系统缓存混放。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from ..config import Config
from ..ingest import resample

__all__ = ["DemucsSeparator", "SeparatorError", "STEM_ORDER"]

# htdemucs 系列固定输出这四条分轨
STEM_ORDER = ("vocals", "drums", "bass", "other")

# 模型可选的权重档位，按「速度 → 质量」排序
MODEL_CHOICES = {
    "htdemucs": "单模型四轨，速度快，默认选择",
    "htdemucs_ft": "四模型精调，质量更好，耗时约 4 倍",
    "hdemucs_mmi": "早期版本，速度与质量居中",
    "mdx_extra": "MDX 架构，质量较高但显存占用大",
}


class SeparatorError(RuntimeError):
    pass


class DemucsSeparator:
    """四轨分离器。模型加载有开销，应在 worker 内复用同一实例。"""

    _model_cache: dict[tuple[str, str], Any] = {}

    def __init__(self, cfg: Config, model_name: str | None = None,
                 device: str | None = None) -> None:
        self.cfg = cfg
        self.model_name = model_name or str(cfg.get("separator.model", "htdemucs"))
        self.device = self._resolve_device(device or cfg.get("separator.device", "auto"))
        self.shifts = int(cfg.get("separator.shifts", 1))
        self.overlap = float(cfg.get("separator.overlap", 0.25))
        self.jobs = int(cfg.get("separator.jobs", 1))
        self.target_sr = int(cfg.get("audio.target_sr", 44100))

        self._point_models_dir()
        self.model = self._load_model()

    # ---------- 准备 ----------

    @staticmethod
    def _resolve_device(want: str) -> str:
        import torch

        if want == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if want == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return want

    def _point_models_dir(self) -> None:
        """把 torch hub 缓存指向项目内 models/，避免污染用户主目录。"""
        models_dir = self.cfg.path("paths.models_dir")
        hub = models_dir / "torch_hub"
        hub.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("TORCH_HOME", str(hub))

        # 关闭 HuggingFace 探测。demucs 在加载模型前会先尝试从
        # huggingface.co 拉取配置文件，本机该域名被路由器 DNS 屏蔽，
        # 每次进程首次加载都要白等约 30 秒重试（实测 5 次重试后放弃）。
        # 而权重实际由 dl.fbaipublicfiles.com 提供，完全不依赖 HF，
        # 因此直接置为离线可省下这段无意义等待。
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    def _load_model(self):
        import torch
        from demucs.pretrained import get_model

        key = (self.model_name, os.environ.get("TORCH_HOME", ""))
        if key in DemucsSeparator._model_cache:
            model = DemucsSeparator._model_cache[key]
        else:
            try:
                model = get_model(self.model_name)
            except Exception as e:
                raise SeparatorError(
                    f"加载 Demucs 模型 {self.model_name} 失败: {e}\n"
                    f"首次运行需要下载权重（约 80 MB/档），"
                    f"权重源 dl.fbaipublicfiles.com 需可访问。"
                )
            DemucsSeparator._model_cache[key] = model

        model.eval()
        model.to(self.device)
        return model

    @property
    def samplerate(self) -> int:
        return int(getattr(self.model, "samplerate", 44100))

    @property
    def sources(self) -> list[str]:
        return list(getattr(self.model, "sources", STEM_ORDER))

    # ---------- 分离 ----------

    def separate(self, audio, sr: int | None = None,
                 progress: bool = False) -> dict[str, np.ndarray]:
        """分离音频。

        audio 可以是 wav 路径或 (channels, samples) 的 float32 数组。
        返回 {分轨名: (channels, samples) float32 数组}，采样率为 target_sr。
        """
        import torch

        data, in_sr = self._load(audio, sr)

        # 重采样到模型要求的采样率
        if in_sr != self.samplerate:
            data = resample(data, in_sr, self.samplerate)

        # 声道数对齐：模型固定要求 2 声道，单声道则复制
        want_ch = int(getattr(self.model, "audio_channels", 2))
        data = self._fit_channels(data, want_ch)

        wav = torch.from_numpy(np.ascontiguousarray(data, dtype=np.float32))
        wav = wav.to(self.device)

        # ---- 以下三步与官方 demucs/separate.py 保持一致 ----
        ref = wav.mean(0)
        wav_n = (wav - ref.mean()) / (ref.std() + 1e-8)

        from demucs.apply import apply_model

        with torch.no_grad():
            out = apply_model(
                self.model,
                wav_n[None],
                device=self.device,
                shifts=self.shifts,
                split=True,
                overlap=self.overlap,
                progress=progress,
                num_workers=self.jobs,
            )
        out = out * ref.std() + ref.mean()
        # ---- 正规化逆变换结束 ----

        arrays = out[0].cpu().numpy()  # (n_sources, channels, samples)

        stems: dict[str, np.ndarray] = {}
        for idx, name in enumerate(self.sources):
            arr = arrays[idx]
            if self.samplerate != self.target_sr:
                arr = resample(arr, self.samplerate, self.target_sr)
            stems[name] = np.ascontiguousarray(arr, dtype=np.float32)

        return stems

    def separate_to_files(self, audio, out_dir: str | Path, sr: int | None = None,
                          stem_names: tuple[str, ...] | None = None) -> dict[str, str]:
        """分离并落盘为 wav，返回 {分轨名: wav 路径}。中间产物留档便于复查。"""
        import soundfile as sf

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stems = self.separate(audio, sr)

        paths: dict[str, str] = {}
        for name, arr in stems.items():
            if stem_names and name not in stem_names:
                continue
            dst = out_dir / f"{name}.wav"
            sf.write(str(dst), np.clip(arr.T, -1.0, 1.0), self.target_sr, subtype="PCM_16")
            paths[name] = str(dst)
        return paths

    # ---------- 辅助 ----------

    def _load(self, audio, sr: int | None) -> tuple[np.ndarray, int]:
        if isinstance(audio, (str, Path)):
            import soundfile as sf

            data, file_sr = sf.read(str(audio), dtype="float32", always_2d=True)
            data = data.T
            if sr is not None and sr != file_sr:
                data = resample(data, file_sr, sr)
                file_sr = sr
            return np.ascontiguousarray(data, dtype=np.float32), int(file_sr)

        arr = np.asarray(audio, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        return np.ascontiguousarray(arr), int(sr or self.target_sr)

    @staticmethod
    def _fit_channels(data: np.ndarray, want: int) -> np.ndarray:
        cur = data.shape[0]
        if cur == want:
            return data
        if cur == 1 and want == 2:
            return np.repeat(data, 2, axis=0)
        if cur > want:
            return data[:want]
        # 声道不足且非单声道情形：循环复制补齐
        reps = -(-want // cur)
        return np.tile(data, (reps, 1))[:want]

    # ---------- 诊断 ----------

    def info(self) -> dict[str, Any]:
        import torch

        return {
            "model": self.model_name,
            "device": self.device,
            "sources": self.sources,
            "samplerate": self.samplerate,
            "audio_channels": int(getattr(self.model, "audio_channels", 2)),
            "shifts": self.shifts,
            "overlap": self.overlap,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "torch_home": os.environ.get("TORCH_HOME", ""),
            "model_choices": MODEL_CHOICES,
        }
