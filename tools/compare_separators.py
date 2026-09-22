"""BS-RoFormer vs htdemucs 分离质量与耗时对比。

没有 ground truth 的情况下，用三个可解释的客观指标：
  1. 重构一致性：各 stem 之和与原混音的误差（衡量分离是否自洽，两者都应低）
  2. 频段纯度：某分轨在其「不应有能量」的频段上的占比
     例：人声轨在 20~80Hz 的能量占比应很低；鼓轨若含大量低频正弦说明串入了贝斯
  3. 分轨间相关性：理想分离下各 stem 应互不相关，相关性越低越好

同时导出 wav，最终判断仍以耳朵为准。
"""

import os
import sys
from pathlib import Path
import time

import numpy as np
import soundfile as sf

ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, ROOT)

# 测试素材从命令行传入，避免把个人下载路径写进公开仓库：
#   python tools/compare_separators.py "某首歌.flac"
SRC = sys.argv[1] if len(sys.argv) > 1 else \
    str(Path(__file__).resolve().parent.parent / "test.flac")
START, DUR = 40.0, 20.0
OUT = os.path.join(ROOT, "work", "sep_compare")
os.makedirs(OUT, exist_ok=True)

from app.config import load_config
from app.ingest import Ingest, resample

cfg = load_config()
ing = Ingest(cfg)
info = ing.prepare(SRC, start=START, duration=DUR)
mono_or_stereo, sr = ing.load_array(info.wav_path, mono=False)
print(f"输入: {DUR:.0f}s @ {sr}Hz  形状 {mono_or_stereo.shape}")


def metrics(stems: dict, mix: np.ndarray, sr: int, tag: str) -> dict:
    """计算三个客观指标。stems 为 {name: (channels, samples)}，mix 同形状。"""
    n = mix.shape[1]
    aligned = {}
    for k, v in stems.items():
        v = np.asarray(v)
        if v.ndim == 1:
            v = v[None, :]
        m = min(n, v.shape[1])
        aligned[k] = v[:, :m]
    if not aligned:
        return {}

    # 1) 重构一致性
    total = np.zeros_like(mix[:, :min(n, min(v.shape[1] for v in aligned.values()))])
    for v in aligned.values():
        total = total + v[:, :total.shape[1]]
    ref = mix[:, :total.shape[1]]
    rec_err = float(np.abs(total - ref).mean() / (np.abs(ref).mean() + 1e-9))

    # 2) 频段纯度：低频占比（20~100Hz）—— 人声/其他轨不应高
    def low_ratio(x):
        X = np.abs(np.fft.rfft(x.mean(axis=0)))
        f = np.fft.rfftfreq(x.shape[1], 1.0 / sr)
        lo = X[(f >= 20) & (f < 100)].sum()
        tot = X[(f >= 20) & (f < 16000)].sum() + 1e-9
        return float(lo / tot)

    # 3) 分轨间相关性（绝对值均值）
    names = list(aligned)
    corrs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = aligned[names[i]].mean(axis=0)
            b = aligned[names[j]].mean(axis=0)
            m = min(len(a), len(b))
            a, b = a[:m], b[:m]
            if a.std() > 0 and b.std() > 0:
                corrs.append(abs(float(np.corrcoef(a, b)[0, 1])))

    return {
        "tag": tag,
        "recon_error": round(rec_err, 4),
        "low_ratio": {k: round(low_ratio(v), 4) for k, v in aligned.items()},
        "mean_abs_corr": round(float(np.mean(corrs)), 4) if corrs else None,
        "rms_db": {k: round(float(20 * np.log10(np.sqrt((v ** 2).mean()) + 1e-9)), 1)
                   for k, v in aligned.items()},
    }


results = {}

# ---------------- htdemucs ----------------
print()
print("=" * 88)
print("htdemucs")
print("=" * 88)
from app.engines.separate import DemucsSeparator

t0 = time.time()
sep_d = DemucsSeparator(cfg, device="cpu")
stems_d = sep_d.separate(mono_or_stereo, sr)
t_d = time.time() - t0
print(f"  耗时 {t_d:.1f}s  实时率 {DUR/t_d:.2f}x  4分钟歌约 {240/DUR*t_d/60:.1f} 分钟")
print(f"  分轨: {sorted(stems_d)}")
results["htdemucs"] = metrics(stems_d, mono_or_stereo, sr, "htdemucs")
for name, arr in stems_d.items():
    sf.write(os.path.join(OUT, f"htdemucs_{name}.wav"),
             np.clip(np.asarray(arr).T, -1, 1), sr, subtype="PCM_16")

# 释放 demucs，避免与 RoFormer 争内存
del sep_d, stems_d
import gc

gc.collect()

# ---------------- BS-RoFormer ----------------
print()
print("=" * 88)
print("BS-RoFormer 4stems (msst)")
print("=" * 88)
from msst.inference import Separator

RF = os.path.join(ROOT, "models", "roformer", "bs_roformer_4stems_ft")
t0 = time.time()
sep_r = Separator(config_path=os.path.join(RF, "config.yaml"),
                  checkpoint_path=os.path.join(RF, "bs_roformer_4stems_ft.pth"),
                  model_type="bs_roformer", force_cpu=True,
                  detailed_progress=False)
print(f"  模型加载完成，设备={sep_r.device}  采样率={sep_r.sample_rate}")
print(f"  分轨名: {sep_r.instruments}")

t1 = time.time()
raw = sep_r.separate(mono_or_stereo, sample_rate=sr)
t_r = time.time() - t1
print(f"  分离耗时 {t_r:.1f}s  实时率 {DUR/t_r:.2f}x  4分钟歌约 {240/DUR*t_r/60:.1f} 分钟")
print(f"  返回类型: {type(raw)}")
if isinstance(raw, dict):
    stems_r = {}
    for k, v in raw.items():
        arr = np.asarray(v)
        if arr.ndim == 1:
            arr = arr[None, :]
        stems_r[k] = arr
        print(f"    {k:<12} shape={arr.shape}  dtype={arr.dtype}")
else:
    arr = np.asarray(raw)
    print(f"    数组 shape={arr.shape}")
    names = list(sep_r.instruments)
    stems_r = {names[i]: arr[..., i, :, :][0] if arr.ndim == 4 else arr[i]
               for i in range(min(len(names), arr.shape[0] if arr.ndim >= 1 else 0))}

results["bs_roformer"] = metrics(stems_r, mono_or_stereo, sr, "bs_roformer")
for name, arr in stems_r.items():
    sf.write(os.path.join(OUT, f"roformer_{name}.wav"),
             np.clip(np.asarray(arr).T, -1, 1), sr, subtype="PCM_16")

# ---------------- 汇总 ----------------
print()
print("=" * 88)
print("对比汇总")
print("=" * 88)
print(f"  耗时：htdemucs {t_d:.1f}s  vs  BS-RoFormer {t_r:.1f}s"
      f"   （慢 {t_r/t_d:.1f} 倍）")
for tag in ("htdemucs", "bs_roformer"):
    m = results.get(tag) or {}
    if not m:
        continue
    print(f"\n  [{tag}]")
    print(f"    重构误差（越低越自洽）: {m.get('recon_error')}")
    print(f"    分轨间平均相关性（越低越干净）: {m.get('mean_abs_corr')}")
    print(f"    各轨 RMS (dB): {m.get('rms_db')}")
    print(f"    各轨 20~100Hz 低频占比（人声/其他应低）:")
    for k, v in (m.get("low_ratio") or {}).items():
        print(f"      {k:<12} {v}")

print()
print(f"试听文件已导出到: {OUT}")
for f in sorted(os.listdir(OUT)):
    print(f"  {f}")
