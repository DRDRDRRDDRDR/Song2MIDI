"""网络：本地代理探测与 HuggingFace 下载。

为什么需要这一层：
    本机路由器 DNS 会把 huggingface.co 解析成 127.0.0.1（DNS 过滤），
    因此直连必然失败。必须走本地代理（Clash 的混合端口），
    由代理侧做远程解析才能绕开污染。

为什么不能依赖环境变量：
    在开发沙箱里 HTTP_PROXY / HTTPS_PROXY 是宿主注入的，端口每次会话都可能变；
    而用户双击运行 exe 时根本没有这些变量。因此程序必须自己探测端口，
    并对候选逐个实测连通性，而不是假定某个端口一定可用。
"""

from __future__ import annotations

import json
import os
import shutil
import ssl
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable

__all__ = ["ProxyProbe", "detect_proxies", "hf_download", "hf_api"]


# 常见的本地代理监听端口。Clash Verge 的混合端口 9549 放在最前，因为本机实测可用。
DEFAULT_PROXY_PORTS = (9549, 33331, 7890, 7897, 10809, 10808, 1080, 2080, 8080, 20171, 8889)

# 用来判断代理是否真的能出网的目标
PROBE_URL = "https://huggingface.co/api/models?limit=1"

_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE


def _registry_proxy() -> str | None:
    """读 Windows 注册表里的手工代理设置（ProxyEnable=1 时 ProxyServer 才有效）。"""
    try:
        import winreg

        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Internet Settings",
        )
        enable = winreg.QueryValueEx(key, "ProxyEnable")[0]
        if not enable:
            return None
        server = winreg.QueryValueEx(key, "ProxyServer")[0]
        if not server:
            return None
        if not server.startswith("http"):
            server = "http://" + server
        return server
    except Exception:
        return None


def detect_proxies(extra_ports: list[int] | None = None) -> list[str]:
    """列出候选代理地址，按可信度排序并去重。

    只做「端口是否在监听」的轻量探测，不做连通性实测 ——
    真正的连通性验证交给 ProxyProbe 按需进行，避免每次启动都等网络超时。
    """
    cands: list[str] = []

    # 1. 环境变量（开发沙箱里就是这个）
    for var in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY"):
        v = os.environ.get(var)
        if v and v.startswith("http"):
            cands.append(v.rstrip("/"))

    # 2. 注册表里的手工代理
    reg = _registry_proxy()
    if reg:
        cands.append(reg.rstrip("/"))

    # 3. 已知端口扫描
    ports = list(extra_ports or []) + list(DEFAULT_PROXY_PORTS)
    for p in ports:
        cands.append(f"http://127.0.0.1:{p}")

    # 去重且保序
    seen: set[str] = set()
    out: list[str] = []
    for c in cands:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _port_listening(proxy: str, timeout: float = 0.6) -> bool:
    import socket

    try:
        parsed = urllib.parse.urlparse(proxy)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 80
    except Exception:
        return False
    s = socket.socket()
    s.settimeout(timeout)
    try:
        s.connect((host, port))
        return True
    except Exception:
        return False
    finally:
        s.close()


class ProxyProbe:
    """探测并缓存一个能真正出网的代理。

    结论会被缓存，避免每次下载都重新实测；缓存有效期默认 10 分钟，
    因为代理端口在沙箱里可能中途变化。
    """

    def __init__(self, extra_ports: list[int] | None = None,
                 probe_url: str = PROBE_URL, ttl: float = 600.0,
                 speed_test_bytes: int = 512 * 1024) -> None:
        self.candidates = detect_proxies(extra_ports)
        self.probe_url = probe_url
        self.ttl = ttl
        self.speed_test_bytes = speed_test_bytes
        self._best: tuple[str | None, float] | None = None
        self.tried: list[dict[str, Any]] = []

    # ---------- 连通性实测 ----------

    @staticmethod
    def _try(url: str, proxy: str | None, timeout: float = 12.0) -> tuple[bool, str]:
        handlers: list[Any] = [urllib.request.HTTPSHandler(context=_SSL_CTX)]
        if proxy:
            handlers.append(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
        else:
            handlers.append(urllib.request.ProxyHandler({}))
        opener = urllib.request.build_opener(*handlers)
        opener.addheaders = [("User-Agent", "Song2MIDI/0.2")]
        try:
            req = urllib.request.Request(url, method="HEAD")
            with opener.open(req, timeout=timeout) as r:
                return True, f"HTTP {r.status}"
        except urllib.error.HTTPError as e:
            # 能拿到状态码就说明网络是通的（403/404 都算通）
            return e.code < 500, f"HTTP {e.code}"
        except Exception as e:
            return False, f"{e.__class__.__name__}: {str(e)[:80]}"

    def _speed(self, url: str, proxy: str | None, timeout: float = 8.0) -> int:
        """小范围下载测速，返回字节/秒；失败返回 -1。

        为什么必须测速：仅判断「通不通」会选出能连但极慢的路径。
        实测本机直连 download.pytorch.org 可达，但速度只有 94 KB/s，
        而走 Clash 代理是 909 KB/s —— 差近 10 倍。若只按连通性选直连，
        2.4 GB 的下载会从 45 分钟变成 7 小时以上。
        """
        handlers: list[Any] = [urllib.request.HTTPSHandler(context=_SSL_CTX)]
        if proxy:
            handlers.append(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
        else:
            handlers.append(urllib.request.ProxyHandler({}))
        opener = urllib.request.build_opener(*handlers)
        opener.addheaders = [("User-Agent", "Song2MIDI/0.2")]
        req = urllib.request.Request(
            url, headers={"Range": f"bytes=0-{self.speed_test_bytes - 1}"})
        try:
            t0 = time.time()
            with opener.open(req, timeout=timeout) as r:
                n = 0
                while n < self.speed_test_bytes:
                    chunk = r.read(65536)
                    if not chunk:
                        break
                    n += len(chunk)
            dt = max(time.time() - t0, 1e-6)
            return int(n / dt)
        except Exception:
            return -1

    def best(self, refresh: bool = False, speed_url: str | None = None) -> str | None:
        """返回可用且最快的代理；若直连最快则返回 None（表示直连）。

        speed_url 为空时只做连通性判断并取第一个可用的，不做测速
        （用于不需要带宽、只求能通的轻量场景）。
        """
        if not refresh and self._best and time.time() - self._best[1] < self.ttl:
            return self._best[0]

        self.tried = []
        results: list[tuple[int, str | None, str]] = []

        ok, detail = self._try(self.probe_url, None)
        self.tried.append({"proxy": "(直连)", "ok": ok, "detail": detail})
        if ok:
            results.append((0, None, "(直连)"))

        for cand in self.candidates:
            if not _port_listening(cand):
                self.tried.append({"proxy": cand, "ok": False, "detail": "端口未监听"})
                continue
            ok, detail = self._try(self.probe_url, cand)
            self.tried.append({"proxy": cand, "ok": ok, "detail": detail})
            if ok:
                results.append((0, cand, cand))

        if not results:
            self._best = (None, time.time())
            return None

        if not speed_url or len(results) == 1:
            chosen = results[0][1]
            self._best = (chosen, time.time())
            return chosen

        # 逐个测速，选最快
        scored: list[tuple[int, str | None, str]] = []
        for _, proxy, label in results:
            sp = self._speed(speed_url, proxy)
            scored.append((sp, proxy, label))
            for t in self.tried:
                if t["proxy"] == label:
                    t["speed_bps"] = sp
        scored.sort(key=lambda x: -x[0])
        chosen = scored[0][1] if scored[0][0] > 0 else results[0][1]
        self._best = (chosen, time.time())
        return chosen

    def build_opener(self, refresh: bool = False, speed_url: str | None = None):
        """构造一个走「可用且最快」路径的 urllib opener。

        传入 speed_url 时会实测各路径速度再选（适合大文件下载）；
        不传则只判连通性，取第一个可用的（适合轻量 API 调用）。
        """
        proxy = self.best(refresh=refresh, speed_url=speed_url)
        handlers: list[Any] = [urllib.request.HTTPSHandler(context=_SSL_CTX)]
        if proxy:
            handlers.append(urllib.request.ProxyHandler({"http": proxy, "https": proxy}))
        else:
            handlers.append(urllib.request.ProxyHandler({}))
        opener = urllib.request.build_opener(*handlers)
        opener.addheaders = [("User-Agent", "Song2MIDI/0.2")]
        return opener, proxy

    def report(self) -> dict[str, Any]:
        return {
            "candidates": self.candidates,
            "attempts": self.tried,
            "chosen": self._best[0] if self._best else None,
        }


# ---------- HuggingFace 下载 ----------

_DEFAULT_PROBE: ProxyProbe | None = None
_PROBE_PORTS: tuple[int, ...] = ()


def _probe(extra_ports: list[int] | None = None) -> ProxyProbe:
    """取进程内单例探测器。

    只有端口列表发生变化时才重建，否则会丢掉已缓存的「哪个代理最快」结论，
    导致每次下载都重新做一轮测速。
    """
    global _DEFAULT_PROBE, _PROBE_PORTS
    ports = tuple(extra_ports or ())
    if _DEFAULT_PROBE is None or (ports and ports != _PROBE_PORTS):
        _DEFAULT_PROBE = ProxyProbe(list(ports) or None)
        _PROBE_PORTS = ports
    return _DEFAULT_PROBE


def hf_api(path: str, timeout: float = 60.0, extra_ports: list[int] | None = None) -> Any:
    """调用 HuggingFace 的 JSON API。"""
    opener, _ = _probe(extra_ports).build_opener()
    url = "https://huggingface.co/api/" + path.lstrip("/")
    with opener.open(url, timeout=timeout) as r:
        return json.load(r)


def hf_download(repo: str, filename: str, dst: str | Path,
                expected_size: int | None = None,
                on_progress: Callable[[int, int], None] | None = None,
                extra_ports: list[int] | None = None,
                retries: int = 3) -> Path:
    """从 HuggingFace 下载文件到本地。

    支持断点续传：若目标文件已存在且小于期望大小，用 Range 续传而不是重下。
    大模型权重动辄数百 MB，网络抖动导致重来一遍的代价很高。
    """
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    url = f"https://huggingface.co/{repo}/resolve/main/{urllib.parse.quote(filename)}"

    for attempt in range(1, retries + 1):
        have = dst.stat().st_size if dst.exists() else 0
        if expected_size and have == expected_size:
            return dst
        if expected_size and have > expected_size:
            dst.unlink()          # 文件比预期还大，说明是脏数据，重来
            have = 0

        opener, proxy = _probe(extra_ports).build_opener(refresh=(attempt > 1))
        headers = {"User-Agent": "Song2MIDI/0.2"}
        if have:
            headers["Range"] = f"bytes={have}-"

        req = urllib.request.Request(url, headers=headers)
        try:
            with opener.open(req, timeout=300) as r:
                mode = "ab" if have and r.status == 206 else "wb"
                if mode == "wb":
                    have = 0
                total = int(r.headers.get("Content-Length") or 0) + have
                with open(dst, mode) as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        have += len(chunk)
                        if on_progress:
                            on_progress(have, total)
            if expected_size and dst.stat().st_size != expected_size:
                raise IOError(
                    f"大小不符：得到 {dst.stat().st_size}，期望 {expected_size}"
                )
            return dst
        except Exception as e:
            if attempt >= retries:
                raise
            time.sleep(2 * attempt)
    return dst
