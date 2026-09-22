"""CI 用：断言安装的 torch 变体与预期一致。

为什么单独写成一个脚本而不是塞进 workflow 的 `python -c`：
  1. PowerShell 的 here-string（`@'...'@`）要求结束符 `'@` **必须顶格**，
     而 YAML 的 `run: |` 块天然带缩进 —— 写进去会静默变成语法错误，
     报错信息完全不提「是缩进问题」。（第一版就是这么失败的。）
  2. 断言逻辑有点长，放脚本里可本地先跑一遍再用。

用法：
    python tools/ci_assert_torch.py cpu
    python tools/ci_assert_torch.py cuda
"""

from __future__ import annotations

import pathlib
import sys

# CI runner（windows-latest）上 Python 的 stdout 编码是 cp1252，不是 UTF-8。
# 直接打印中文会抛 UnicodeEncodeError 并让脚本崩在第一行 —— 断言逻辑根本
# 执行不到、却看起来像「断言失败」。这里强制 UTF-8，同时输出仍保持 ASCII，
# 双保险。
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def main() -> int:
    want = (sys.argv[1] if len(sys.argv) > 1 else "").strip().lower()
    if want not in ("cpu", "cuda"):
        print(f"usage: {sys.argv[0]} cpu|cuda")
        return 2

    import torch

    v = torch.__version__
    lib = pathlib.Path(torch.__file__).parent / "lib"
    files = list(lib.glob("*")) if lib.is_dir() else []
    total_mb = sum(p.stat().st_size for p in files if p.is_file()) / 2**20
    dlls = [p.name for p in files if p.suffix.lower() == ".dll"]
    cuda_dlls = [n for n in dlls
                 if any(k in n.lower() for k in ("cuda", "cudnn", "cublas", "cufft", "curand"))]

    print(f"  want variant : {want}")
    print(f"  torch        : {v}")
    print(f"  torch/lib    : {len(dlls)} DLLs, {total_mb:.1f} MB")
    print(f"  CUDA DLLs    : {len(cuda_dlls)}")
    if cuda_dlls:
        print(f"    e.g.       : {', '.join(cuda_dlls[:4])}")

    problems = []
    if want == "cuda":
        if "+cu" not in v:
            problems.append(f"version '{v}' has no '+cu' -> CPU build installed")
        if not cuda_dlls:
            problems.append("no CUDA DLL in torch/lib")
        if total_mb < 1500:
            problems.append(f"torch/lib only {total_mb:.0f} MB, CUDA build should exceed 1500 MB")
    else:
        if "+cu" in v:
            problems.append(f"version '{v}' has '+cu' -> CUDA build installed")
        if total_mb > 800:
            problems.append(f"torch/lib is {total_mb:.0f} MB, CPU build should be far smaller")

    if problems:
        print()
        print("  ASSERTION FAILED:")
        for p in problems:
            print(f"    - {p}")
        print()
        print("  Likely cause: install order. If torch is installed BEFORE")
        print("  'pip install -r requirements-build.txt', pip re-resolves deps")
        print("  (demucs requires torch>=2.1) and pulls the CPU build from PyPI.")
        print("  Correct order: requirements first, then torch with --index-url.")
        return 1

    print()
    print(f"  ASSERTION PASSED: {want} variant is correct")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
