"""实测 BS-RoFormer 在 GPU 上的推理速度，验证「质量优先」方案是否真的可用。

CPU 上官方推理流程实测 0.05x 实时（4 分钟的歌约 79 分钟）。
装 CUDA 后需要确认提速幅度，才能判断该方案是否兑现。
"""

import os
import sys
from pathlib import Path
import time

import numpy as np

ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, ROOT)

from app.config import load_config
from app.engines.separate_roformer import RoFormerSeparator

cfg = load_config()
RF = os.path.join(ROOT, "models", "roformer", "bs_roformer_4stems_ft")

print("=" * 84)
print("BS-RoFormer 设备与速度实测")
print("=" * 84)

t0 = time.time()
sep = RoFormerSeparator(cfg, model_dir=RF, device="cuda")
print(f"模型加载耗时 {time.time()-t0:.1f}s")
for k, v in sep.info().items():
    print(f"  {k}: {v}")

print()
print("=" * 84)
print("推理速度")
print("=" * 84)
sr = 44100
for dur in (10.0, 30.0):
    n = int(dur * sr)
    audio = (np.random.randn(2, n) * 0.1).astype(np.float32)
    t1 = time.time()
    out = sep.separate(audio, sr)
    dt = time.time() - t1
    stems = {k: v.shape for k, v in out.items()}
    per4 = 240.0 / (dur / dt)
    print(f"  {dur:5.1f}s 音频 → 耗时 {dt:7.2f}s  实时率 {dur/dt:6.2f}x  "
          f"→ 4 分钟歌约 {per4/60:5.2f} 分钟")
    print(f"        分轨: {stems}")

print()
print("=" * 84)
print("对照")
print("=" * 84)
print("  CPU 实测 0.05x 实时（4 分钟歌约 79 分钟）")
print("  若 GPU 实时率 > 1.0x，则 4 分钟歌在 4 分钟内完成，方案成立")
