"""命令行入口：批量把歌曲文件转成多轨 MIDI。

    python tools/run_cli.py "歌1.flac" "歌2.mp3" -o out
    python tools/run_cli.py "D:\\Music" --jobs 3
    python tools/run_cli.py "歌.flac" --no-separate          # 只出单轨旋律，跳过分离（快）
    python tools/run_cli.py "歌.flac" --start 40 --duration 30   # 先取片段试跑
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import load_config
from app.ingest import SUPPORTED_EXTS, Ingest
from app.pipeline import PipelineOptions, Song2MidiPipeline
from app.pool import ParallelRunner

sys.path.insert(0, str(ROOT / "app"))


def collect_inputs(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS:
                    out.append(f)
        elif p.is_file():
            out.append(p)
        else:
            print(f"  ! 路径不存在，已跳过: {p}")
    # 去重并保持顺序
    seen = set()
    uniq: list[Path] = []
    for f in out:
        key = str(f.resolve()).lower()
        if key not in seen:
            seen.add(key)
            uniq.append(f)
    return uniq


def fmt_sec(s: float) -> str:
    s = int(round(s))
    return f"{s // 60}:{s % 60:02d}"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="歌曲文件转多轨 MIDI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("inputs", nargs="+", help="音频文件或目录")
    ap.add_argument("-o", "--out", default=None, help="输出目录（默认取配置的 out/）")
    ap.add_argument("--jobs", type=int, default=None, help="文件级并发数")
    ap.add_argument("--sep-jobs", type=int, default=None, help="分离阶段并发数")
    ap.add_argument("--tr-jobs", type=int, default=None, help="转录阶段并发数")
    ap.add_argument("--separation", default=None, choices=["auto", "force", "skip"],
                    help="分离策略：auto=自动检测无鼓无贝斯的独奏曲（默认）| "
                         "force=强制四轨分离 | skip=跳过分离整首直转")
    ap.add_argument("--no-separate", action="store_true",
                    help="等价于 --separation skip（兼容旧用法）")
    ap.add_argument("--force-separate", action="store_true",
                    help="等价于 --separation force")
    ap.add_argument("--no-drums", action="store_true", help="跳过鼓组转录")
    ap.add_argument("--no-chords", action="store_true", help="跳过和弦识别")
    ap.add_argument("--stems", default="vocals,drums,bass,other",
                    help="要转录的分轨，逗号分隔")
    ap.add_argument("--engine", default=None, choices=["demucs", "roformer"],
                    help="分离引擎：demucs（快）| roformer（质量更好，需 GPU）")
    ap.add_argument("--model", default=None,
                    help="Demucs 档位：htdemucs / htdemucs_ft / hdemucs_mmi / mdx_extra")
    ap.add_argument("--device", default=None, help="auto / cuda / cpu")
    ap.add_argument("--start", type=float, default=None, help="起始秒（只处理片段）")
    ap.add_argument("--duration", type=float, default=None, help="片段时长秒")
    ap.add_argument("--no-quantize", action="store_true", help="关闭节拍量化")
    ap.add_argument("--list", action="store_true", help="只列出将处理的文件，不实际执行")
    args = ap.parse_args()

    cfg = load_config()
    if args.out:
        cfg._data.setdefault("paths", {})["out_dir"] = args.out
    if args.engine:
        cfg._data.setdefault("separator", {})["engine"] = args.engine
    if args.model:
        cfg._data.setdefault("separator", {})["model"] = args.model
    if args.device:
        cfg._data.setdefault("separator", {})["device"] = args.device

    files = collect_inputs(args.inputs)
    if not files:
        print("未找到任何可处理的音频文件。")
        return 1

    print("=" * 72)
    print(f"待处理 {len(files)} 个文件")
    print("=" * 72)
    for i, f in enumerate(files, 1):
        try:
            mb = f.stat().st_size / 1048576
            print(f"  {i:>3}. {f.name}  ({mb:.1f} MB)")
        except OSError:
            print(f"  {i:>3}. {f.name}")
    if args.list:
        return 0

    # 分离策略三态：None=自动检测，True=强制分离，False=跳过分离
    if args.no_separate or args.separation == "skip":
        sep_mode = False
    elif args.force_separate or args.separation == "force":
        sep_mode = True
    else:
        sep_mode = None

    opts = PipelineOptions(
        separate=sep_mode,
        transcribe_drums=not args.no_drums,
        transcribe_chords=not args.no_chords,
        stems=tuple(s.strip() for s in args.stems.split(",") if s.strip()),
        start=args.start,
        duration=args.duration,
        quantize=False if args.no_quantize else None,
    )

    pipeline = Song2MidiPipeline(cfg)
    runner = ParallelRunner(cfg, file_workers=args.jobs,
                            sep_workers=args.sep_jobs, tr_workers=args.tr_jobs)

    print(f"\n并行配置：文件级 {runner.file_workers} / "
          f"分离 {runner.sep_workers} / 转录 {runner.tr_workers}")
    if opts.separate is False:
        print("分离引擎：已跳过（整首作为单轨转录）")
    else:
        sep = pipeline.separator
        mode = "（自动检测：独奏曲会跳过）" if opts.separate is None else ""
        print(f"分离引擎：{sep.model_name} @ {sep.device}{mode}")
    print()

    t0 = time.perf_counter()

    def on_event(event: str, data: dict) -> None:
        if event == "job_done":
            res = data["result"]
            tag = "完成" if res["ok"] else "失败"
            line = (f"[{data['done']}/{data['total']}] {tag} "
                    f"{Path(res['source']).name}  "
                    f"耗时 {res['elapsed_sec']:.1f}s")
            if res["ok"] and res.get("speedup_vs_realtime"):
                line += f"  ({res['speedup_vs_realtime']:.1f}x 实时)"
            print(line)
            if not res["ok"]:
                print(f"        错误: {res['error']}")
            elif res.get("payload", {}).get("summary", {}).get("warnings"):
                for w in res["payload"]["summary"]["warnings"]:
                    print(f"        提示: {w}")
        elif event == "finish":
            pass

    tasks = [(f.stem, pipeline.make_task(f, opts)) for f in files]
    results = runner.run(tasks, on_event=on_event)

    total = time.perf_counter() - t0
    ok = [r for r in results if r.ok]

    print("\n" + "=" * 72)
    print(f"全部完成：成功 {len(ok)} / 共 {len(results)}，总耗时 {fmt_sec(total)}")
    print("=" * 72)

    for r in sorted(results, key=lambda x: x.job_id):
        if not r.ok:
            continue
        stem_stats = r.payload.get("summary", {}).get("stems", [])
        print(f"\n{r.job_id}  ({r.duration_sec:.0f}s 音频, 耗时 {r.elapsed_sec:.1f}s)")
        tempo = r.payload.get("summary", {}).get("tempo", {})
        print(f"  节拍: {tempo.get('bpm')} BPM, 相位 {tempo.get('phase_sec')}s "
              f"(置信度 {tempo.get('confidence')})")
        for s in stem_stats:
            print(f"  {s['stem']:<8} {s.get('count', 0):>5} 音符   "
                  f"音域 {s.get('pitch_min_name', '-')}–{s.get('pitch_max_name', '-')}   "
                  f"平均时值 {s.get('duration_mean_ms', '-')} ms")
        for f in r.payload.get("export", {}).get("midi_files", []):
            print(f"  → {f}")
        # 回读校验：写文件成功不等于文件正确，把实际音符数读出来给用户看
        for v in r.payload.get("export", {}).get("verification", []):
            if v.get("ok"):
                print(f"  [OK] 校验 {Path(v['file']).name}: "
                      f"{v['instruments']} 音轨 / {v['total_notes']} 音符 / "
                      f"{v['duration_sec']}s / {v['size_bytes']:,} 字节")
                for t in v.get("tracks", []):
                    drum = " [鼓轨]" if t.get("is_drum") else ""
                    print(f"      {t['name']:<28} {t['notes']:>5} 音符  "
                          f"GM {t['program']}{drum}")
            else:
                print(f"  [失败] 校验 {v.get('file')}: {v.get('error')}")

    if runner.stats():
        print("\n并发统计：")
        print(json.dumps(runner.stats(), ensure_ascii=False, indent=2))

    return 0 if len(ok) == len(results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
