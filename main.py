"""Song2MIDI 入口。

默认启动桌面界面（PySide6）。
    python main.py                     启动桌面界面
    python main.py --cli 歌.flac ...   命令行模式，参数原样透传给 tools/run_cli.py
    python main.py --web               启动本地 Web 服务（备用入口）
    python main.py --check             只跑环境自检

打包成 exe 后双击即用；把音频文件直接拖到 exe 图标上也会走批量处理。

关于打包：PyInstaller 冻结后 sys.executable 指向 exe 自身，而 tools/*.py
是源码文件、不再存在于冻结环境中。因此各分支优先在进程内直接调用，
只有在源码环境下才用子进程方式，避免打包后功能失效。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Windows 控制台默认编码是 GBK(936)，无法表示 ✓ ✗ ▸ 这类符号。
# 一旦 print 到它们就抛 UnicodeEncodeError，而且会中断整个流程 ——
# 实测：CLI 模式下 Demucs 分离、四轨转录、MIDI 写出全部成功，
# 只在打印「✓ 校验」这一行时崩掉，用户看到的是「跑到最后报编码错」。
# 这里把编码错误降级为替换字符，保证输出永远不会成为流程的失败点。
# 同时各脚本里也已改用 GBK 可表示的符号，这里是兜底。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except Exception:
        pass

# PyInstaller 会设置该属性，用于判断是否处于冻结环境
FROZEN = getattr(sys, "frozen", False)


def _run_script(rel: str, argv: list[str]) -> int:
    """运行 tools/ 下的脚本。

    冻结环境下 tools/ 不再是可执行文件，改为导入其 main() 在进程内执行。
    """
    if FROZEN:
        name = Path(rel).stem
        mod = __import__(f"tools.{name}", fromlist=["main"])
        saved = sys.argv
        try:
            sys.argv = [name, *argv]
            return int(mod.main() or 0)
        finally:
            sys.argv = saved
    # 源码模式下走这里；同样要抑制黑窗（本程序通常是 GUI 方式启动的）
    kwargs = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    return subprocess.call([sys.executable, str(ROOT / rel), *argv], **kwargs)


def _has_gui() -> bool:
    try:
        import PySide6  # noqa: F401

        return True
    except ImportError:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(add_help=True, description="歌曲文件转多轨 MIDI")
    ap.add_argument("--gui", action="store_true", help="启动桌面界面（默认行为）")
    ap.add_argument("--web", action="store_true", help="启动本地 Web 服务")
    ap.add_argument("--cli", action="store_true", help="命令行模式")
    ap.add_argument("--check", action="store_true", help="只运行环境自检")
    # 用 parse_known_args 而不是 nargs="*" 的位置参数：
    # 位置参数接不住 `--start 40` 这类选项，argparse 会直接报
    # "unrecognized arguments"。CLI 的子命令参数必须原样透传。
    ns, passthrough = ap.parse_known_args()

    if ns.check:
        # 透传其余参数，使 `--check --json` 能拿到结构化输出 ——
        # 界面把自检放到子进程里跑（避免抢主线程的 GIL 与 CPU），
        # 主进程靠解析这份 JSON 回填 torch / CUDA 状态。
        return _run_script("tools/check_env.py", passthrough)

    # 拖放文件到 exe 图标上时，Windows 会把路径作为位置参数传入。
    # 处理方式：启动桌面界面并把文件预填进去、自动开始，而不是走静默批处理。
    # 原因是 exe 以 GUI 子系统构建（无控制台），批处理模式下用户看不到
    # 任何进度与结果，体验上像是「双击了没反应」。
    positional = [a for a in passthrough if not a.startswith("-")]

    if ns.cli:
        return _run_script("tools/run_cli.py", passthrough)

    if ns.web:
        from app.api import serve

        return serve()

    if not _has_gui():
        print("未检测到 PySide6，无法启动桌面界面。")
        print("安装：pip install PySide6")
        print("或改用命令行模式：python main.py --cli 歌.flac")
        return 1

    if positional:
        # 只保留真实存在的文件，避免把误传的参数当输入
        files = [a for a in positional if Path(a).exists()]
        if files:
            from app.gui import run_gui

            return run_gui(initial_files=files, autorun=True)

    from app.gui import run_gui

    return run_gui()


if __name__ == "__main__":
    code = main()

    # 冻结环境下任务完成后进程不肯退出。
    #
    # 实测：`Song2MIDI.exe --cli 歌.flac --no-separate` 已经打印完
    # 「全部完成：成功 1 / 共 1」并生成了 MIDI，但进程持续存活、150 秒
    # 后仍未结束 —— 原因是 onnxruntime / numba 之类的库会创建非守护线程，
    # CPython 3.9+ 在解释器退出时会 join 所有线程，于是卡在这里。
    # 源码环境没有这个问题，只在打包后出现。
    #
    # 用户侧的观感是「程序跑完了但不结束、像卡死」。因此冻结环境下
    # 任务结束即显式退出进程，跳过线程清理（此时已无待写入状态）。
    if FROZEN:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        os._exit(code)

    raise SystemExit(code)
