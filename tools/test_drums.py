"""鼓组转录引擎的定量验证。

方法：合成一段鼓点，音高与时刻都完全已知，再让引擎去识别，比对
「检测率 / 分类准确率」。合成信号有明确答案，因此可以给出真实指标，
而不是靠耳朵判断「好像还行」。

    python tools/test_drums.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import load_config
from app.engines.drums import DrumTranscriber


def _kind_of(pitch: int, mapping: dict[str, int]) -> str:
    """把输出 MIDI 音高反查回鼓件名，用于评估分类是否正确。"""
    for kind, p in mapping.items():
        if int(p) == int(pitch):
            return kind
    return f"pitch{pitch}"


def bandpass(x: np.ndarray, sr: int, lo: float, hi: float) -> np.ndarray:
    """理想带通（FFT 频域掩码）。

    之前用一阶差分模拟高通，那会让能量随频率线性上升，合成出的军鼓
    频谱重心高得不真实，等于给分类器出了个偏题。改用频域掩码后，
    合成信号的频谱分布才与实际鼓件相符。
    """
    n = len(x)
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    spec[(freqs < lo) | (freqs > hi)] = 0.0
    return np.fft.irfft(spec, n=n).astype(np.float32)


def _normalize(x: np.ndarray) -> np.ndarray:
    peak = float(np.abs(x).max())
    return (x / peak).astype(np.float32) if peak > 0 else x


def synth_kick(sr: int, dur: float = 0.25) -> np.ndarray:
    """底鼓：140Hz 迅速下滑到 45Hz 的正弦 + 快速衰减包络。能量集中在 20~150Hz。"""
    n = int(sr * dur)
    t = np.linspace(0, dur, n, endpoint=False)
    freq = 140 * np.exp(-t * 22) + 45
    phase = 2 * np.pi * np.cumsum(freq) / sr
    env = np.exp(-t * 18)
    return _normalize(np.sin(phase) * env * 0.95)


def synth_snare(sr: int, dur: float = 0.25) -> np.ndarray:
    """军鼓：200~7000Hz 宽带噪声 + 190Hz 固有振荡。频谱重心约 1kHz 量级。"""
    n = int(sr * dur)
    t = np.linspace(0, dur, n, endpoint=False)
    rng = np.random.default_rng(42)
    noise = bandpass(rng.normal(0, 1, n).astype(np.float32), sr, 200.0, 7000.0)
    noise = _normalize(noise)
    body = np.sin(2 * np.pi * 190 * t).astype(np.float32)
    env = np.exp(-t * 16)
    return _normalize((noise * 0.7 + body * 0.3) * env)


def synth_hat(sr: int, open_: bool = False) -> np.ndarray:
    """踩镲：7000~17000Hz 窄带高频噪声。开镲衰减明显更慢。"""
    dur = 0.45 if open_ else 0.09
    n = int(sr * dur)
    t = np.linspace(0, dur, n, endpoint=False)
    rng = np.random.default_rng(7 if open_ else 3)
    noise = bandpass(rng.normal(0, 1, n).astype(np.float32), sr, 7000.0, 17000.0)
    noise = _normalize(noise)
    env = np.exp(-t * (9 if open_ else 60))
    return _normalize(noise * env * 0.5)


def main() -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", action="store_true", help="打印每个 onset 的特征与候选")
    ap.add_argument("--debug-limit", type=int, default=16)
    args = ap.parse_args()

    cfg = load_config()
    sr = int(cfg.get("audio.target_sr", 44100))
    tr = DrumTranscriber(cfg)

    # ---- 构造已知答案的鼓点 ----
    bpm = 120.0
    beat = 60.0 / bpm
    total = 8.0
    mix = np.zeros(int(sr * total), dtype=np.float32)

    truth: list[tuple[float, str]] = []

    def place(sig: np.ndarray, at: float, kind: str) -> None:
        i = int(at * sr)
        end = min(len(mix), i + len(sig))
        if end > i:
            mix[i:end] += sig[: end - i]
            truth.append((round(at, 4), kind))

    # 每拍底鼓；每 2、4 拍加军鼓；八分音符闭镲；第 4 拍开镲
    for b in range(int(total / beat)):
        t = b * beat
        place(synth_kick(sr), t, "kick")
        if b % 2 == 1:
            place(synth_snare(sr), t, "snare")
        for off in (0.0, beat / 2):
            if abs(off) > 1e-9 or True:
                kind = "open_hat" if (b % 4 == 3 and off == 0.0) else "closed_hat"
                place(synth_hat(sr, open_=(kind == "open_hat")), t + off, kind)

    mix = np.clip(mix, -1.0, 1.0)
    print("=" * 68)
    print("鼓组转录引擎验证（合成信号，答案已知）")
    print("=" * 68)
    print(f"音频: {total:.1f}s, {sr} Hz")
    print(f"真实击打总数: {len(truth)}")
    from collections import Counter
    print(f"  分布: {dict(Counter(k for _, k in truth))}")

    result = tr.transcribe(mix, sr)

    if args.debug:
        a = tr.analyze(mix, sr, limit=args.debug_limit)
        print("\n--- 中间量 ---")
        print(json.dumps(a["stats"], ensure_ascii=False, indent=2))
        print("\n--- 每个 onset 的特征与候选 ---")
        for d in a["detections"]:
            print(f"  t={d['time']:6.3f}s  rel_e={d['rel_energy']:5.2f}  "
                  f"decay={d['decay_ms']:6.0f}ms  主导段={d['dominant_band']}")
            print(f"      norm={d['norm']}")
            print(f"      flux={d['flux']}")
            print(f"      → candidates={d['candidates']}")

    print(f"\n检测到: {len(result.notes)} 个音符")
    print(f"引擎元信息: peaks={result.meta['peaks_detected']}, "
          f"接受={result.meta['hits_accepted']}, 拒绝={result.meta['hits_rejected']}")
    print(f"分类计数: {result.meta['counts_by_kind']}")

    # ---- 评估 ----
    # 鼓组转录是多标签问题：同一时刻可能有多个鼓件同时击打。
    # 因此不能按「最近时间逐个配对」来比对 —— 同拍三个鼓的时间几乎相同，
    # 贪心配对会把集合内的配对顺序搞错，把正确的分类算成错误。
    # 正确做法是把结果按 onset 时刻聚成簇，再逐簇做集合比对。
    def cluster(events: list[tuple[float, str]], gap: float = 0.04
                ) -> list[tuple[float, set[str]]]:
        """把 (时间, 类别) 按时间间隔聚簇，返回 [(代表时刻, 类别集合)]。"""
        if not events:
            return []
        ev = sorted(events, key=lambda x: x[0])
        out: list[list] = [[ev[0][0], {ev[0][1]}]]
        for t, k in ev[1:]:
            if t - out[-1][0] <= gap:
                out[-1][1].add(k)
            else:
                out.append([t, {k}])
        return [(c[0], c[1]) for c in out]

    truth_clusters = cluster(truth)
    det_clusters = cluster([(n.start, _kind_of(n.pitch, tr.mapping)) for n in result.notes])
    tol = 0.06

    used_det: set[int] = set()
    exact = 0
    per_kind = {k: {"tp": 0, "fp": 0, "fn": 0} for k in set(k for _, k in truth)}
    matched_truth_clusters = 0

    for tt, tset in truth_clusters:
        best_i, best_d = None, 1e9
        for i, (dt, _ds) in enumerate(det_clusters):
            if i in used_det:
                continue
            d = abs(dt - tt)
            if d < best_d:
                best_i, best_d = i, d
        if best_i is None or best_d > tol:
            for k in tset:
                per_kind.setdefault(k, {"tp": 0, "fp": 0, "fn": 0})["fn"] += 1
            continue
        used_det.add(best_i)
        matched_truth_clusters += 1
        dset = det_clusters[best_i][1]
        if dset == tset:
            exact += 1
        for k in tset:
            per_kind.setdefault(k, {"tp": 0, "fp": 0, "fn": 0})
            per_kind[k]["tp" if k in dset else "fn"] += 1
        for k in dset - tset:
            per_kind.setdefault(k, {"tp": 0, "fp": 0, "fn": 0})["fp"] += 1

    for i, (dt, ds) in enumerate(det_clusters):
        if i not in used_det:
            for k in ds:
                per_kind.setdefault(k, {"tp": 0, "fp": 0, "fn": 0})["fp"] += 1

    n_truth = len(truth_clusters)
    n_det = len(det_clusters)
    cluster_recall = matched_truth_clusters / n_truth if n_truth else 0
    cluster_precision = len(used_det) / n_det if n_det else 0
    set_accuracy = exact / n_truth if n_truth else 0

    print("\n--- 评估（多标签，按 onset 簇做集合比对，±60ms 容差）---")
    print(f"  真实击打簇: {n_truth}   检出簇: {n_det}")
    print(f"  簇召回率  : {cluster_recall*100:5.1f}%  ({matched_truth_clusters}/{n_truth})")
    print(f"  簇精确率  : {cluster_precision*100:5.1f}%")
    print(f"  鼓件集合完全正确: {set_accuracy*100:5.1f}%  ({exact}/{n_truth})")
    print("\n  分鼓件指标:")
    print(f"    {'鼓件':<12}{'命中':>5}{'漏检':>6}{'误报':>6}{'召回':>8}{'精确':>8}")
    for k in sorted(per_kind):
        s = per_kind[k]
        rec = s["tp"] / (s["tp"] + s["fn"]) if (s["tp"] + s["fn"]) else 0.0
        pre = s["tp"] / (s["tp"] + s["fp"]) if (s["tp"] + s["fp"]) else 0.0
        print(f"    {k:<12}{s['tp']:>5}{s['fn']:>6}{s['fp']:>6}"
              f"{rec*100:>7.1f}%{pre*100:>7.1f}%")

    print("\n--- 前 16 个检测结果 ---")
    for n in result.sorted_notes()[:16]:
        from app.notes import pitch_name
        print(f"  {n.start:7.3f}s  pitch={n.pitch:>3} {pitch_name(n.pitch):<4} "
              f"vel={n.velocity}")

    ok = cluster_recall > 0.85 and set_accuracy > 0.7
    print("\n" + "=" * 68)
    print("判定：" + ("通过" if ok else "未达标，需要调整分类规则或参数"))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
