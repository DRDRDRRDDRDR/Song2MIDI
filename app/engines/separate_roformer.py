"""BS-RoFormer 四轨音源分离（基于 msst 的官方推理流程）。

为什么用 msst 而不是 lucidrains 的 bs-roformer：
    两者虽然都叫 BS-RoFormer，但架构代际不同。实测用 lucidrains 的实现
    加载本 checkpoint（ZFTurbo/MSST 框架训练的）只有 32% 的权重能对上，
    mask estimator 的张量尺寸差 2 倍。而 msst 的实现可以 100% 严格加载
    （1355/1355 张量，missing=0 / unexpected=0）。

    判定依据很直接：模型参数量 131,704,164，fp32 即 502.3 MB，
    与权重文件 502.82 MB 吻合。

性能（本机实测，20 秒片段）：
    CPU  0.05x 实时 → 4 分钟的歌约 79 分钟（官方流程含 4 倍重叠）
    GPU  待实测；这是本模型在 CPU 上不实用的根本原因。

许可证：本模型 SYH99999/bs_roformer_4stems_ft 为 Apache-2.0。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from ..config import Config

__all__ = ["RoFormerSeparator", "RoFormerError"]

# msst 支持的 model_type 取值中与本项目相关的几个
SUPPORTED_MODEL_TYPES = ("bs_roformer", "mel_band_roformer", "mdemucs4ht", "demucs4ht", "mdx23c")


class RoFormerError(RuntimeError):
    pass


class RoFormerSeparator:
    """BS-RoFormer 四轨分离器。接口与 DemucsSeparator 保持一致，便于互换。"""

    _cache: dict[tuple[str, str], Any] = {}

    def __init__(self, cfg: Config, model_dir: str | Path | None = None,
                 device: str | None = None, model_type: str | None = None) -> None:
        self.cfg = cfg
        self.device = self._resolve_device(device or cfg.get("separator.device", "auto"))
        self.model_type = model_type or str(
            cfg.get("separator.roformer_type", "bs_roformer"))
        self.model_dir = Path(model_dir) if model_dir else self._default_dir()
        self.checkpoint_name = str(
            cfg.get("separator.roformer_checkpoint", "bs_roformer_4stems_ft.pth"))
        self.config_name = str(cfg.get("separator.roformer_config", "config.yaml"))

        self._ensure_files()
        self._sep = self._load()

    # ---------- 准备 ----------

    @staticmethod
    def _resolve_device(want: str) -> str:
        import torch

        if want == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if want == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return want

    def _default_dir(self) -> Path:
        models_dir = self.cfg.path("paths.models_dir")
        return models_dir / "roformer" / "bs_roformer_4stems_ft"

    def _ensure_files(self) -> None:
        """确认权重与配置就位；缺失时按需从 HuggingFace 下载。

        exe 里会内置这两份文件；但仍然保留自动下载作为兜底 ——
        文件损坏、被误删、或用户想换模型时都能自愈。
        """
        cfg_path = self.model_dir / self.config_name
        ckpt_path = self.model_dir / self.checkpoint_name

        if cfg_path.is_file() and ckpt_path.is_file() and ckpt_path.stat().st_size > 1024:
            return

        repo = str(self.cfg.get("separator.roformer_repo",
                                "SYH99999/bs_roformer_4stems_ft"))
        expected = self.cfg.get("separator.roformer_checkpoint_bytes", 527245586)
        # 代理端口从配置读取，允许用户按自己的环境调整
        ports = self.cfg.get("network.proxy_ports")
        try:
            from ..net import hf_download

            self.model_dir.mkdir(parents=True, exist_ok=True)
            if not cfg_path.is_file():
                hf_download(repo, self.config_name, cfg_path, extra_ports=ports)
            if not (ckpt_path.is_file() and ckpt_path.stat().st_size == expected):
                hf_download(repo, self.checkpoint_name, ckpt_path,
                            expected_size=int(expected) if expected else None,
                            extra_ports=ports)
        except Exception as e:
            raise RoFormerError(
                f"BS-RoFormer 权重缺失且自动下载失败：{e}\n"
                f"期望位置：{ckpt_path}\n"
                f"若本机 huggingface.co 不可直连，程序会自动探测本地代理端口；"
                f"也可以手动把权重放到上述路径。"
            )

    def _load(self):
        try:
            from msst.inference import Separator as _MsstSeparator
        except ImportError as e:
            raise RoFormerError(
                f"未安装 msst，无法使用 BS-RoFormer 分离。\n"
                f"安装：pip install msst omegaconf ml_collections matplotlib\n"
                f"原始错误：{e}"
            )

        key = (str(self.model_dir), self.device)
        if key in RoFormerSeparator._cache:
            return RoFormerSeparator._cache[key]

        # msst 内部会自行判断设备；CUDA 可用时用 0 号卡
        kwargs: dict[str, Any] = {
            "config_path": str(self.model_dir / self.config_name),
            "checkpoint_path": str(self.model_dir / self.checkpoint_name),
            "model_type": self.model_type,
            "detailed_progress": False,
        }
        if self.device == "cpu":
            kwargs["force_cpu"] = True
        else:
            kwargs["device_ids"] = 0

        try:
            sep = _MsstSeparator(**kwargs)
        except Exception as e:
            raise RoFormerError(
                f"加载 BS-RoFormer 失败：{e.__class__.__name__}: {e}"
            ) from e

        RoFormerSeparator._cache[key] = sep
        return sep

    # ---------- 属性 ----------

    @property
    def model_name(self) -> str:
        return f"{self.model_type}({self.checkpoint_name.replace('.pth', '')})"

    @property
    def samplerate(self) -> int:
        try:
            return int(self._sep.sample_rate)
        except Exception:
            return 44100

    @property
    def sources(self) -> list[str]:
        try:
            return list(self._sep.instruments)
        except Exception:
            return ["drums", "bass", "other", "vocals"]

    # ---------- 分离 ----------

    def separate(self, audio, sr: int | None = None,
                 progress: bool = False) -> dict[str, np.ndarray]:
        """分离音频，返回 {分轨名: (channels, samples) float32}。

        接口与 DemucsSeparator.separate 一致，管线可无缝互换。
        """
        data, in_sr = self._load_array(audio, sr)

        try:
            raw = self._sep.separate(data, sample_rate=in_sr, channels_first=True)
        except Exception as e:
            raise RoFormerError(f"分离执行失败：{e.__class__.__name__}: {e}") from e

        if not isinstance(raw, dict):
            arr = np.asarray(raw)
            names = self.sources
            out: dict[str, np.ndarray] = {}
            if arr.ndim == 4:            # (stems, channels, samples) 或 (1, stems, ch, s)
                a = arr[0] if arr.shape[0] == 1 else arr
                for i, n in enumerate(names[:a.shape[0]]):
                    out[n] = np.ascontiguousarray(a[i], dtype=np.float32)
            else:
                for i, n in enumerate(names[:arr.shape[0]]):
                    out[n] = np.ascontiguousarray(arr[i], dtype=np.float32)
            return out

        stems: dict[str, np.ndarray] = {}
        for name, v in raw.items():
            a = np.asarray(v, dtype=np.float32)
            if a.ndim == 1:
                a = a[None, :]
            stems[str(name)] = np.ascontiguousarray(a)
        return stems

    def _load_array(self, audio, sr: int | None) -> tuple[np.ndarray, int]:
        from ..ingest import resample

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
        return np.ascontiguousarray(arr), int(sr or self.samplerate)

    # ---------- 诊断 ----------

    def info(self) -> dict[str, Any]:
        import torch

        ckpt = self.model_dir / self.checkpoint_name
        return {
            "engine": "roformer",
            "model": self.model_name,
            "model_type": self.model_type,
            "device": self.device,
            "sources": self.sources,
            "samplerate": self.samplerate,
            "model_dir": str(self.model_dir),
            "checkpoint": str(ckpt),
            "checkpoint_mb": round(ckpt.stat().st_size / 1048576, 1) if ckpt.is_file() else None,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "license": "Apache-2.0 (SYH99999/bs_roformer_4stems_ft)",
            "note": "官方推理流程含 4 倍重叠，CPU 上约 0.05x 实时，强烈建议使用 GPU",
        }
