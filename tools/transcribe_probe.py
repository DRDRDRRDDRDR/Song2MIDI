"""转录链路快速验证工具。

截取真实歌曲的一个片段，跑完整解码 → 转录 → 音符流程，打印结果统计。
用途：换模型、改阈值、升级依赖后，用它确认链路仍然正确。

    python tools/transcribe_probe.py "歌曲路径" --start 30 --duration 60
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import load_config
from app.ingest import Ingest
from app.notes import StemNotes
from app.engines.pitch_onnx import PitchTranscriber


def histogram(notes, bins: int = 12) -> str:
    """按音高画一个粗略分布直方图，用于一眼判断音域是否合理。"""
    if not notes:
        return "(无音符)"
    pitches = [n.pitch for n in notes]
    lo, hi = min(pitches), max(pitches)
    if hi == lo:
        return f"{lo} 单一音高 x{len(pitches)}"
    width = max(1, (hi - lo + 1) // bins)
    lines = []
    for b in range(lo, hi + 1, width):
        cnt = sum(1 for p in pitches if b <= p < b + width)
        if cnt:
            bar = "#" * min(60, cnt)
            lines.append(f"  {b:>3}-{min(b+width-1, hi):>3} |{bar} {cnt}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", help="输入音频路径")
    ap.add_argument("--start", type=float, default=None, help="起始秒")
    ap.add_argument("--duration", type=float, default=None, help="截取时长秒")
    ap.add_argument("--stem", default="mix", help="分轨名，用于读取 per_stem 参数")
    ap.add_argument("--poly", type=int, default=None, help="覆盖最大复音数")
    ap.add_argument("--no-melodia", action="store_true", help="关闭 melodia 后处理")
    args = ap.parse_args()

    cfg = load_config()
    print("=" * 68)
    print("转录链路验证")
    print("=" * 68)

    t0 = time.perf_counter()
    ing = Ingest(cfg)
    raw_info = ing.inspect(args.audio)
    print(f"输入      : {Path(args.audio).name}")
    print(f"容器/编码 : {raw_info.get('container')} / {raw_info.get('codec')}")
    print(f"采样率    : {raw_info.get('sample_rate')} Hz, {raw_info.get('channels')} 声道, "
          f"{raw_info.get('bits_per_sample') or '?'} bit")
    print(f"时长      : {raw_info.get('hms')} ({raw_info.get('duration_sec'):.2f} s)")
    for w in raw_info.get("warnings") or []:
        print(f"  ! {w}")

    info = ing.prepare(args.audio, start=args.start, duration=args.duration)
    print(f"解码产物  : {info.wav_path}")
    print(f"  {info.duration_sec:.2f} s ({'缓存命中' if info.from_cache else '新解码'})")
    print(f"解码耗时  : {time.perf_counter() - t0:.2f} s")

    t1 = time.perf_counter()
    tr = PitchTranscriber(cfg)
    print(f"\n模型      : {tr.model_path} ({tr.model_path.stat().st_size:,} 字节)")
    print(f"执行后端  : {tr.providers}")

    stem_cfg = cfg.section("transcription").get("per_stem", {}).get(args.stem, {})
    print(f"分轨参数  : stem={args.stem} {stem_cfg or '(使用全局默认)'}")

    result: StemNotes = tr.transcribe(
        info.wav_path,
        stem=args.stem,
        max_polyphony=args.poly,
        melodia_trick=False if args.no_melodia else None,
    )
    elapsed = time.perf_counter() - t1

    print(f"\n推理耗时  : {elapsed:.2f} s "
          f"(音频 {info.duration_sec:.1f} s, 实时率 {info.duration_sec / max(elapsed, 1e-6):.1f}x)")

    print("\n--- 引擎元信息 ---")
    for k, v in result.meta.items():
        print(f"  {k}: {v}")

    print("\n--- 结果统计 ---")
    for k, v in result.stats().items():
        print(f"  {k}: {v}")

    print("\n--- 音高分布 ---")
    print(histogram(result.notes))

    print("\n--- 前 20 个音符 ---")
    for n in result.sorted_notes()[:20]:
        print(f"  {n.start:8.3f} → {n.end:8.3f}  ({n.duration*1000:6.0f} ms)  "
              f"{n.pitch:>3} {n.name:<4} vel={n.velocity}")

    ok = len(result.notes) > 0
    print("\n" + "=" * 68)
    print("结论：" + (f"链路正常，提取到 {len(result.notes)} 个音符" if ok else "链路跑通但未提取到音符，需检查阈值"))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
