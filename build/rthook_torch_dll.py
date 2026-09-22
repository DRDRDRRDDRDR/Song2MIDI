"""PyInstaller 运行时 hook：把 torch 的 DLL 目录注册进搜索路径。

为什么需要这个 hook：
    torch 的 C 扩展（`torch/_C.cp3xx-win_amd64.pyd`）依赖同目录树下的
    `torch/lib/` 里的 `c10.dll`、`torch_cpu.dll`、`torch_cuda.dll`、
    `cublas64_12.dll`、`cudnn*64_9.dll` 等。而 Windows 加载扩展模块时，
    只在该扩展自身所在目录与系统路径里找依赖，**不会自动搜索 torch/lib**。

    结果是冻结后 `import torch` 抛 `OSError: [WinError 126] 找不到指定的模块`，
    而这个报错信息完全不提缺的是哪个 DLL，极难定位 ——
    本项目的 exe 就曾因此表现为「torch 未安装」，尽管 `_internal/torch/lib`
    里 37 个 DLL 一个不少。

    解法是在导入 torch 之前，用 os.add_dll_directory 显式注册这些目录。

放在运行时 hook 而不是业务代码里，是为了保证在任何 import torch 之前执行 ——
包括第三方库（msst / demucs / basic-pitch）自己触发的导入。
"""

import os
import sys
from pathlib import Path

if getattr(sys, "frozen", False):
    base = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))

    # 顺序有讲究：torch/lib 必须最先，其余作为兜底
    candidates = [
        base / "torch" / "lib",
        base / "torch",
        base / "shiboken6",
        base,
    ]

    for d in candidates:
        if not d.is_dir():
            continue
        try:
            os.add_dll_directory(str(d))
        except (OSError, AttributeError):
            # 非 Windows 或目录不可用时忽略
            pass
        # PATH 也一并加上：部分 DLL 仍走传统搜索顺序
        os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")
