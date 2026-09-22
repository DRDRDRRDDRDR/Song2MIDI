"""按 URL 清单下载文件，支持断点续传。

为什么不用 pip 直接装：
    CUDA 版 torch 单文件 2.4 GB，而 pip 不支持断点续传。网络抖动一次就要重来，
    代价过高。本脚本用 HTTP Range 续传，中断后重跑即可接着下。

    python tools/fetch_wheels.py models/wheels/urls.txt
"""

from __future__ import annotations

import os
import ssl
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.net import ProxyProbe  # noqa: E402

_CTX = ssl.create_default_context()
_CTX.check_hostname = False
_CTX.verify_mode = ssl.CERT_NONE


def probe_size(opener, url: str) -> int | None:
    try:
        req = urllib.request.Request(url, method="HEAD")
        with opener.open(req, timeout=60) as r:
            n = r.headers.get("Content-Length")
            return int(n) if n else None
    except Exception:
        return None


def fetch(opener, url: str, dst: Path, retries: int = 6) -> bool:
    name = urllib.parse.unquote(url.split("/")[-1].split("#")[0])
    if not dst.suffix:
        dst = dst / name
    dst.parent.mkdir(parents=True, exist_ok=True)

    total = probe_size(opener, url)
    print(f"\n> {name}")
    print(f"  远端大小: {total/1048576:.1f} MB" if total else "  远端大小: 未知")

    for attempt in range(1, retries + 1):
        have = dst.stat().st_size if dst.exists() else 0
        if total and have == total:
            print(f"  已完成（{have/1048576:.1f} MB），跳过")
            return True
        if total and have > total:
            print("  本地文件大于远端，视为脏数据，重下")
            dst.unlink()
            have = 0

        headers = {"User-Agent": "Song2MIDI/0.2"}
        if have:
            headers["Range"] = f"bytes={have}-"
            print(f"  第 {attempt} 次尝试，从 {have/1048576:.1f} MB 续传")
        else:
            print(f"  第 {attempt} 次尝试，从头下载")

        try:
            req = urllib.request.Request(url, headers=headers)
            t0 = time.time()
            with opener.open(req, timeout=300) as r:
                mode = "ab" if have and r.status == 206 else "wb"
                if mode == "wb":
                    have = 0
                cl = r.headers.get("Content-Length")
                expected = (int(cl) + have) if cl else total
                last_report = 0.0
                with open(dst, mode) as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        have += len(chunk)
                        now = time.time()
                        if now - last_report > 5:
                            last_report = now
                            pct = f"{have/expected*100:5.1f}%" if expected else "  ?  "
                            rate = have / max(now - t0 + 1e-6, 1e-6) / 1048576
                            print(f"    {pct}  {have/1048576:8.1f} MB  {rate:5.2f} MB/s",
                                  flush=True)

            size = dst.stat().st_size
            if total and size != total:
                raise IOError(f"大小不符：{size} != {total}")
            print(f"  完成  {size/1048576:.1f} MB  用时 {time.time()-t0:.0f}s")
            return True
        except Exception as e:
            print(f"  失败: {e.__class__.__name__}: {str(e)[:140]}")
            if attempt < retries:
                wait = min(20, 3 * attempt)
                print(f"  {wait}s 后重试...")
                time.sleep(wait)

    print("  多次重试后仍未成功")
    return False


def main() -> int:
    list_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "models" / "wheels" / "urls.txt"
    dst_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else list_path.parent

    urls = [u.strip() for u in list_path.read_text(encoding="utf-8").splitlines() if u.strip()]
    if not urls:
        print(f"清单为空: {list_path}")
        return 1

    print("=" * 80)
    print("探测可用路径并测速（大文件必须按速度选，不能只看通不通）")
    print("=" * 80)
    probe = ProxyProbe()
    # 用第一个 URL 做测速基准
    opener, proxy = probe.build_opener(speed_url=urls[0])
    print(f"  选中: {proxy or '(直连)'}")
    for a in probe.tried:
        mark = "OK  " if a["ok"] else "失败"
        sp = a.get("speed_bps")
        sp_txt = f"  {sp/1048576:6.2f} MB/s" if sp and sp > 0 else ""
        print(f"    {mark} {a['proxy']:<28} {a['detail']}{sp_txt}")

    ok = 0
    for u in urls:
        if fetch(opener, u, dst_dir):
            ok += 1

    print()
    print("=" * 80)
    print(f"完成 {ok}/{len(urls)}")
    for f in sorted(dst_dir.iterdir()):
        if f.is_file() and f.suffix == ".whl":
            print(f"  {f.stat().st_size/1048576:9.1f} MB  {f.name}")
    return 0 if ok == len(urls) else 2


if __name__ == "__main__":
    raise SystemExit(main())
