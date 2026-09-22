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


def main() -> int:
    want = (sys.argv[1] if len(sys.argv) > 1 else "").strip().lower()
    if want not in ("cpu", "cuda"):
        print(f"用法: {sys.argv[0]} cpu|cuda")
        return 2

    import torch

    v = torch.__version__
    lib = pathlib.Path(torch.__file__).parent / "lib"
    files = list(lib.glob("*")) if lib.is_dir() else []
    total_mb = sum(p.stat().st_size for p in files if p.is_file()) / 2**20
    dlls = [p.name for p in files if p.suffix.lower() == ".dll"]
    cuda_dlls = [n for n in dlls
                 if any(k in n.lower() for k in ("cuda", "cudnn", "cublas", "cufft", "curand"))]

    print(f"  期望变体 : {want}")
    print(f"  torch    : {v}")
    print(f"  torch/lib: {len(dlls)} 个 DLL，共 {total_mb:.1f} MB")
    print(f"  CUDA DLL : {len(cuda_dlls)} 个")
    if cuda_dlls:
        print(f"    例如   : {', '.join(cuda_dlls[:4])}")

    problems = []
    if want == "cuda":
        if "+cu" not in v:
            problems.append(f"版本号 '{v}' 不含 '+cu'，说明装的是 CPU 版")
        if not cuda_dlls:
            problems.append("torch/lib 里没有任何 CUDA DLL")
        if total_mb < 1500:
            problems.append(f"torch/lib 仅 {total_mb:.0f} MB，CUDA 版应超过 1500 MB")
    else:
        if "+cu" in v:
            problems.append(f"版本号 '{v}' 含 '+cu'，说明装的是 CUDA 版")
        if total_mb > 800:
            problems.append(f"torch/lib 达 {total_mb:.0f} MB，CPU 版应远小于此")

    if problems:
        print()
        print("  断言失败：")
        for p in problems:
            print(f"    · {p}")
        print()
        print("  常见原因：安装顺序不对 —— 先装了 torch，随后 pip 处理其它包的")
        print("  依赖时从默认源重新解析，把 torch 换成了另一个变体。")
        print("  正确顺序：先 pip install -r requirements-build.txt，最后再装 torch。")
        return 1

    print()
    print(f"  断言通过：{want} 变体正确")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
