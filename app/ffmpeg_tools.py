"""ffmpeg / ffprobe 定位与调用。

设计约束（本项目硬性要求）：
1. 不要求用户安装 ffmpeg，不修改系统 PATH，不写注册表。
2. 优先复用本机已有的 ffmpeg 可执行文件；显式配置优先级最高。
3. 所有音频格式差异统一在 ingest 层抹平，下游引擎只面对标准 wav。

优先级：config 显式路径 > 环境变量 > PATH > 常见安装位置 > 受限深度递归搜索。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

__all__ = [
    "FFmpegNotFound",
    "FFmpegError",
    "FFmpegTools",
    "find_ffmpeg",
    "find_ffprobe",
    "get_tools",
]

_EXE = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
_PROBE_EXE = "ffprobe.exe" if os.name == "nt" else "ffprobe"

# 用户主目录。用 Path.home() 展开，不把具体用户名写进源码 ——
# 这个文件会随仓库公开，硬编码 C:\Users\<name>\... 会泄露本机信息，
# 而且换台机器就失效。
_HOME = Path.home()

# 调用外部控制台程序时抑制窗口。
#
# 为什么必须有：本程序以 GUI 子系统构建（`console=False`），自身没有控制台。
# Windows 下这种进程去调用 ffmpeg / ffprobe 这类控制台程序时，系统会**为每个
# 子进程新建一个控制台窗口** —— 用户就会看到黑窗不停闪现。处理一首歌要调
# 2~3 次（探测格式 → 解码为 wav → 读时长），所以是「几个黑窗」。
# 非 Windows 平台没有这个概念，返回空字典即可。
_NO_WINDOW_KWARGS: dict = (
    {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
)

# 已知的常见安装位置。逐个探测，命中即用。
_WELL_KNOWN_DIRS: tuple[str, ...] = (
    str(_HOME / "Desktop" / "MUSICBOT"),
    r"C:\Program Files\iTubeGo",
    r"C:\Program Files\M3U8-Downloader\resources\node_modules\ffmpeg-static",
    r"C:\ffmpeg\bin",
    r"C:\Program Files\ffmpeg\bin",
    r"C:\ProgramData\chocolatey\bin",
    str(_HOME / "scoop" / "shims"),
)

# 受限递归搜索的根目录（仅在已知位置全部落空时启用）
_SEARCH_ROOTS: tuple[str, ...] = (
    str(_HOME),
    r"C:\tools",
    r"C:\Program Files",
)

_MAX_SEARCH_DEPTH = 5
_SKIP_DIR_NAMES = {
    "node_modules", ".git", "__pycache__", "site-packages", "venv", ".venv",
    "cache", "AppData", "Windows", "System32", "Packages",
}


class FFmpegNotFound(RuntimeError):
    """在所有候选位置都未能找到 ffmpeg / ffprobe。"""


class FFmpegError(RuntimeError):
    """ffmpeg 进程以非零状态退出。"""

    def __init__(self, cmd: list[str], returncode: int, stderr: str) -> None:
        self.cmd = cmd
        self.returncode = returncode
        self.stderr = stderr
        tail = stderr.strip().splitlines()[-6:]
        super().__init__(
            f"ffmpeg 退出码 {returncode}\n命令: {' '.join(cmd)}\n" + "\n".join(tail)
        )


def _iter_candidate_dirs() -> Iterable[Path]:
    """产出可能含 ffmpeg.exe 的目录，顺序即优先级。"""
    for raw in _WELL_KNOWN_DIRS:
        p = Path(raw)
        if p.is_dir():
            yield p
            # 发行包常多套一层版本目录，补一层通配
            try:
                for sub in sorted(p.iterdir()):
                    if sub.is_dir():
                        yield sub
                        yield from (d for d in sub.iterdir() if d.is_dir())
            except (OSError, PermissionError):
                continue


def _bounded_search(exe_name: str) -> str | None:
    """受限深度递归搜索。仅在已知位置全部落空时调用。"""
    for root_raw in _SEARCH_ROOTS:
        root = Path(root_raw)
        if not root.is_dir():
            continue
        base_depth = len(root.parts)
        for dirpath, dirnames, filenames in os.walk(root):
            cur = Path(dirpath)
            if len(cur.parts) - base_depth > _MAX_SEARCH_DEPTH:
                dirnames[:] = []
                continue
            dirnames[:] = [d for d in dirnames if d not in _SKIP_DIR_NAMES]
            if exe_name in filenames:
                return str(cur / exe_name)
    return None


def _probe_dir_for(exe_name: str, dirs: Iterable[Path]) -> str | None:
    for d in dirs:
        cand = d / exe_name
        if cand.is_file():
            return str(cand)
        # 部分发行包把可执行文件放在 bin/ 子目录
        cand_bin = d / "bin" / exe_name
        if cand_bin.is_file():
            return str(cand_bin)
    return None


def _resolve(exe_name: str, explicit: str | None, env_var: str) -> str:
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            raise FFmpegNotFound(f"配置中指定的路径不存在: {explicit}")
        return str(p)

    env_val = os.environ.get(env_var)
    if env_val and Path(env_val).is_file():
        return env_val

    on_path = shutil.which(exe_name)
    if on_path:
        return on_path

    hit = _probe_dir_for(exe_name, _iter_candidate_dirs())
    if hit:
        return hit

    hit = _bounded_search(exe_name)
    if hit:
        return hit

    raise FFmpegNotFound(
        f"未找到 {exe_name}。请在 config.yaml 的 ffmpeg.{env_var} 中显式指定完整路径。"
    )


def find_ffmpeg(explicit: str | None = None) -> str:
    """定位 ffmpeg 可执行文件。"""
    return _resolve(_EXE, explicit, "FFMPEG_BINARY")


def find_ffprobe(explicit: str | None = None) -> str:
    """定位 ffprobe。找不到时退化为使用 ffmpeg 本身做媒体探测。"""
    try:
        return _resolve(_PROBE_EXE, explicit, "FFPROBE_BINARY")
    except FFmpegNotFound:
        ff = find_ffmpeg()
        sibling = Path(ff).with_name(_PROBE_EXE)
        if sibling.is_file():
            return str(sibling)
        raise


@dataclass
class FFmpegTools:
    """一对 ffmpeg / ffprobe 封装，附带能力自检。"""

    ffmpeg: str
    ffprobe: str
    version: str = ""

    # ---------- 构造 ----------

    @classmethod
    def create(cls, ffmpeg: str | None = None, ffprobe: str | None = None) -> "FFmpegTools":
        ff = find_ffmpeg(ffmpeg)
        fp = find_ffprobe(ffprobe)
        tools = cls(ffmpeg=ff, ffprobe=fp)
        tools.version = tools._read_version()
        return tools

    def _read_version(self) -> str:
        try:
            out = self.run_raw([self.ffmpeg, "-hide_banner", "-version"], timeout=30)
            return out.splitlines()[0] if out else "unknown"
        except Exception:
            return "unknown"

    # ---------- 底层调用 ----------

    def run_raw(self, cmd: list[str], timeout: int | None = None) -> str:
        """执行命令并返回 stdout。非零退出抛 FFmpegError。

        Windows 下 ffmpeg 输出为 UTF-8，显式指定避免 GBK 解码乱码。
        """
        proc = subprocess.run(
            cmd,
            capture_output=True,
            timeout=timeout,
            # 关键：让子进程输出可被正确解码，规避中文路径乱码
            encoding="utf-8",
            errors="replace",
            # 抑制黑窗（见 _NO_WINDOW_KWARGS 的说明）
            **_NO_WINDOW_KWARGS,
        )
        if proc.returncode != 0:
            raise FFmpegError(cmd, proc.returncode, proc.stderr or "")
        return proc.stdout or ""

    def run(self, args: list[str], timeout: int | None = None) -> str:
        """以 ffmpeg 执行参数列表（自动前置 -hide_banner -nostdin）。"""
        return self.run_raw([self.ffmpeg, "-hide_banner", "-nostdin", *args], timeout=timeout)

    # ---------- 媒体探测 ----------

    def probe(self, src: str) -> dict[str, Any]:
        """读取音频基本信息：时长、采样率、声道数、编码格式。"""
        raw = self.run_raw([
            self.ffprobe, "-v", "error",
            "-show_entries", "format=duration,bit_rate,format_name",
            "-show_entries", "stream=codec_name,codec_type,sample_rate,channels,bits_per_raw_sample",
            "-of", "json", str(src),
        ], timeout=120)
        data = json.loads(raw)

        fmt = data.get("format", {}) or {}
        streams = [s for s in (data.get("streams") or []) if s.get("codec_type") == "audio"]
        if not streams:
            raise FFmpegError([self.ffprobe, src], 0, "文件中未找到音频流")
        st = streams[0]

        def _num(key: str) -> float | None:
            v = fmt.get(key)
            try:
                return float(v) if v is not None else None
            except (TypeError, ValueError):
                return None

        sr_raw = st.get("sample_rate")
        ch_raw = st.get("channels")
        bits_raw = st.get("bits_per_raw_sample")
        return {
            "path": str(src),
            "duration_sec": _num("duration"),
            "bit_rate": _num("bit_rate"),
            "container": fmt.get("format_name"),
            "codec": st.get("codec_name"),
            "sample_rate": int(sr_raw) if sr_raw else None,
            "channels": int(ch_raw) if ch_raw else None,
            "bits_per_sample": int(bits_raw) if bits_raw else None,
            "n_audio_streams": len(streams),
        }

    # ---------- 解码 ----------

    def decode_to_wav(
        self,
        src: str,
        dst: str,
        target_sr: int = 44100,
        channels: int = 2,
        subtype: str = "PCM_16",
        start: float | None = None,
        duration: float | None = None,
    ) -> str:
        """把任意容器/编码解码为标准 wav。

        统一转成 PCM wav 的理由：下游引擎（Demucs / ONNX 推理）对
        mp3/flac/m4a 的支持路径各不相同，先归一再喂进去，可消除
        解码差异带来的不确定性，也便于复现问题。
        """
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        args: list[str] = []
        if start is not None:
            args += ["-ss", f"{start:.6f}"]
        args += ["-i", str(src)]
        if duration is not None:
            args += ["-t", f"{duration:.6f}"]
        args += [
            "-vn",                      # 丢弃封面图等视频流
            "-map", "0:a:0",            # 只取第一条音频流
            "-ac", str(channels),
            "-ar", str(target_sr),
            "-c:a", "pcm_s16le" if subtype == "PCM_16" else "pcm_s24le",
            "-f", "wav",
            "-y", str(dst),
        ]
        self.run(args)
        if not Path(dst).is_file():
            raise FFmpegError([self.ffmpeg, *args], 0, f"输出文件未生成: {dst}")
        return str(dst)

    def extract_segment(self, src: str, dst: str, start: float, duration: float,
                        target_sr: int = 44100, channels: int = 2) -> str:
        """截取片段。用于快速试样，避免对整首歌反复跑重模型。"""
        return self.decode_to_wav(src, dst, target_sr=target_sr, channels=channels,
                                  start=start, duration=duration)

    # ---------- 能力自检 ----------

    def capabilities(self) -> dict[str, bool]:
        """检查编码器/解码器是否齐备。用于启动时自检，提前暴露问题。"""
        enc = self.run(["-encoders"], timeout=60)
        dec = self.run(["-decoders"], timeout=60)
        want_enc = ["libmp3lame", "libvorbis", "libopus", "aac", "flac", "pcm_s16le"]
        want_dec = ["mp3", "flac", "vorbis", "aac", "opus"]
        result = {f"enc:{c}": c in enc for c in want_enc}
        result.update({f"dec:{c}": (" " + c + " ") in dec for c in want_dec})
        return result


_TOOLS_CACHE: FFmpegTools | None = None


def get_tools(ffmpeg: str | None = None, ffprobe: str | None = None,
              refresh: bool = False) -> FFmpegTools:
    """进程内单例，避免重复探测磁盘。"""
    global _TOOLS_CACHE
    if _TOOLS_CACHE is None or refresh or ffmpeg or ffprobe:
        _TOOLS_CACHE = FFmpegTools.create(ffmpeg, ffprobe)
    return _TOOLS_CACHE
