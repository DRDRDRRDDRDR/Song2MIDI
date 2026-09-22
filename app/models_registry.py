"""模型清单与下载：模型不再随软件打包，改由这里按需拉取与下载。

## 为什么模型外置

三个模型合计约 750 MB，而 CUDA 版 torch 本身已有 2.5 GB —— 打在一起会让产物
膨胀到 5 GB，超过 GitHub Release 的 2 GB 单文件上限，也没法常规分发。
外置之后软件本体变小、模型可跨版本复用、用户按需下载。

## 清单为什么放 GitHub

清单是**数据**而非代码。放在仓库里，改一个 JSON 就能让所有已安装的软件
看到新模型，**无需重新发版**。软件点「刷新」即拉取。

## 镜像选择（实测数据，别拍脑袋改）

    镜像                直连        走代理
    gh-proxy.com        663 ms      284 ms      ← 最快
    cdn.jsdelivr.net    637 ms      1716 ms     ← 直连快
    ghproxy.net         894 ms      1414 ms
    raw.githubusercontent 13083 ms   超时        ← 千万别作首选

`raw.githubusercontent.com` 在国内直连要 13 秒、走代理直接超时 —— 若把它排在
第一位，用户会以为界面卡死了。因此按上表顺序尝试，且每个镜像只等 8 秒。
"""

from __future__ import annotations

import json
import os
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

__all__ = [
    "ModelSpec", "ModelFile", "load_manifest", "model_status",
    "download_model", "installed_summary", "MANIFEST_URL", "REPO",
]

REPO = "DRDRDRRDDRDR/Song2MIDI"
BRANCH = "main"
RAW_PATH = f"{REPO}/{BRANCH}/models.json"
MANIFEST_URL = f"https://raw.githubusercontent.com/{RAW_PATH}"

# 按实测速度排序；raw.githubusercontent 放最后兜底。
# 注意 ghproxy.net 已实测对 raw 路径返回 404（它的 URL 格式与 gh-proxy 不同），
# 所以不要按「名字看着像」就加进来 —— 每个镜像都要实测过。
MANIFEST_MIRRORS = [
    f"https://gh-proxy.com/https://raw.githubusercontent.com/{RAW_PATH}",
    f"https://cdn.jsdelivr.net/gh/{REPO}@{BRANCH}/models.json",
    f"https://raw.githubusercontent.com/{RAW_PATH}",
]

MIRROR_TIMEOUT = 8          # 单镜像超时（秒）——短一些，失败就换
CACHE_MAX_AGE = 24 * 3600   # 本地缓存有效期


@dataclass
class ModelFile:
    """模型由若干文件组成（如 htdemucs_ft 是 4 个子模型）。"""

    target: str                                  # 相对 models_root 的路径
    size_mb: float = 0.0
    sources: list[str] = field(default_factory=list)


@dataclass
class ModelSpec:
    id: str
    name: str
    desc: str = ""
    tags: list[str] = field(default_factory=list)
    size_mb: float = 0.0
    required: bool = False
    recommended: bool = False
    license: str = ""
    engine: str = ""            # demucs | roformer | piano
    model_arg: str = ""         # 传给引擎的档位名
    files: list[ModelFile] = field(default_factory=list)

    @property
    def total_mb(self) -> float:
        return sum(f.size_mb for f in self.files) or self.size_mb


def _ssl_ctx() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _opener(proxy: str | None = None, use_proxy: bool = True):
    hs = [urllib.request.HTTPSHandler(context=_ssl_ctx())]
    if use_proxy:
        from .net import _probe  # 复用项目已有的代理探测

        try:
            op, _ = _probe(None).build_opener()
            return op
        except Exception:
            pass
    hs.append(urllib.request.ProxyHandler({}))
    op = urllib.request.build_opener(*hs)
    return op


def _cache_path() -> Path:
    from .config import MODELS_ROOT

    return MODELS_ROOT.parent / "manifest_cache.json"


def _builtin_manifest() -> dict[str, Any]:
    """离线回退清单。

    与仓库里的 models.json 保持同步即可；全部镜像都不可达时用它，
    保证界面不会空着（用户仍能看到有哪些模型、能按缓存里的地址下载）。
    """
    here = Path(__file__).resolve().parent
    for cand in (here / "models_builtin.json",):
        try:
            if cand.is_file():
                return json.loads(cand.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"manifest_version": 0, "models": []}


def load_manifest(force: bool = False, on_status: Callable[[str], None] | None = None
                  ) -> tuple[dict[str, Any], str]:
    """拉取模型清单。返回 (清单, 来源说明)。

    顺序：本地缓存（未过期且非强制）→ 各镜像 → 内置回退。
    """
    def say(msg: str) -> None:
        if on_status:
            try:
                on_status(msg)
            except Exception:
                pass

    cache = _cache_path()

    if not force and cache.is_file():
        try:
            age = time.time() - cache.stat().st_mtime
            if age < CACHE_MAX_AGE:
                data = json.loads(cache.read_text(encoding="utf-8"))
                say(f"用本地缓存（{age/3600:.1f} 小时前）")
                return data, f"本地缓存"
        except Exception:
            pass

    last_err = ""
    for i, url in enumerate(MANIFEST_MIRRORS, 1):
        host = urllib.parse.urlparse(url).netloc
        say(f"拉取清单 {i}/{len(MANIFEST_MIRRORS)}：{host}")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Song2MIDI"})
            with _opener().open(req, timeout=MIRROR_TIMEOUT) as r:
                raw = r.read()
            data = json.loads(raw.decode("utf-8"))
            if not isinstance(data.get("models"), list):
                raise ValueError("清单格式不对：缺少 models 列表")
            try:
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_bytes(raw)
            except OSError:
                pass
            say(f"成功：{len(data['models'])} 个模型")
            return data, host
        except Exception as e:
            last_err = f"{host}: {e.__class__.__name__}"
            continue

    say(f"所有镜像均失败（{last_err}），使用内置清单")
    return _builtin_manifest(), "内置清单"


def parse_models(manifest: dict[str, Any]) -> list[ModelSpec]:
    out: list[ModelSpec] = []
    for m in manifest.get("models") or []:
        files = []
        for f in m.get("files") or []:
            files.append(ModelFile(
                target=str(f.get("target", "")),
                size_mb=float(f.get("size_mb", 0) or 0),
                sources=[str(s) for s in (f.get("sources") or [])],
            ))
        out.append(ModelSpec(
            id=str(m.get("id", "")),
            name=str(m.get("name", m.get("id", ""))),
            desc=str(m.get("desc", "")),
            tags=[str(t) for t in (m.get("tags") or [])],
            size_mb=float(m.get("size_mb", 0) or 0),
            required=bool(m.get("required")),
            recommended=bool(m.get("recommended")),
            license=str(m.get("license", "")),
            engine=str(m.get("engine", "")),
            model_arg=str(m.get("model_arg", "")),
            files=files,
        ))
    return out


def model_status(spec: ModelSpec, models_root: Path) -> dict[str, Any]:
    """检查模型在本地的状态。"""
    done = 0
    have_mb = 0.0
    missing: list[str] = []
    for f in spec.files:
        p = models_root / Path(*f.target.replace("\\", "/").split("/"))
        if p.is_file() and p.stat().st_size > 1024:
            done += 1
            have_mb += p.stat().st_size / 1048576
        else:
            missing.append(f.target)
    return {
        "id": spec.id,
        "installed": done == len(spec.files) and len(spec.files) > 0,
        "partial": 0 < done < len(spec.files),
        "files_done": done,
        "files_total": len(spec.files),
        "have_mb": round(have_mb, 1),
        "missing": missing,
    }


def download_file(target: Path, sources: list[str], expected_mb: float = 0.0,
                  on_progress: Callable[[int, int, str], None] | None = None,
                  retries: int = 3) -> tuple[bool, str]:
    """从多个源下载单个文件，支持断点续传。返回 (成功, 说明)。"""
    target.parent.mkdir(parents=True, exist_ok=True)
    expected = int(expected_mb * 1048576) if expected_mb else 0

    for src in sources:
        host = urllib.parse.urlparse(src).netloc
        for attempt in range(1, retries + 1):
            have = target.stat().st_size if target.is_file() else 0
            if expected and have >= expected * 0.99:
                return True, f"已完成（{have/1048576:.1f} MB）"
            if expected and have > expected * 1.5:
                target.unlink()      # 明显偏大，是脏数据
                have = 0
            try:
                headers = {"User-Agent": "Song2MIDI"}
                if have:
                    headers["Range"] = f"bytes={have}-"
                req = urllib.request.Request(src, headers=headers)
                with _opener().open(req, timeout=60) as r:
                    total = int(r.headers.get("Content-Length") or 0) + have
                    mode = "ab" if have and getattr(r, "status", 200) == 206 else "wb"
                    got = have if mode == "ab" else 0
                    with open(target, mode) as fh:
                        while True:
                            chunk = r.read(262144)
                            if not chunk:
                                break
                            fh.write(chunk)
                            got += len(chunk)
                            if on_progress:
                                try:
                                    on_progress(got, total, host)
                                except Exception:
                                    pass
                size = target.stat().st_size
                if expected and size < expected * 0.99:
                    continue      # 没下完，换下一轮（会走续传）
                return True, f"{size/1048576:.1f} MB"
            except Exception as e:
                if attempt == retries:
                    break
                time.sleep(1.5)
    return False, f"所有源均失败（{len(sources)} 个）"


def download_model(spec: ModelSpec, models_root: Path,
                   on_progress: Callable[[int, int, str], None] | None = None,
                   on_status: Callable[[str], None] | None = None) -> tuple[bool, str]:
    """下载一个模型的全部文件。返回 (成功, 说明)。"""
    total_files = len(spec.files)
    for i, f in enumerate(spec.files, 1):
        dst = models_root / Path(*f.target.replace("\\", "/").split("/"))
        if dst.is_file() and dst.stat().st_size > 1024 and \
                (not f.size_mb or dst.stat().st_size >= f.size_mb * 1048576 * 0.99):
            if on_status:
                on_status(f"[{i}/{total_files}] 已存在，跳过 {Path(f.target).name}")
            continue
        if on_status:
            on_status(f"[{i}/{total_files}] 下载 {Path(f.target).name}"
                      f"（约 {f.size_mb:.0f} MB）")
        ok, msg = download_file(dst, f.sources, f.size_mb, on_progress)
        if not ok:
            return False, f"{Path(f.target).name}: {msg}"
        if on_status:
            on_status(f"[{i}/{total_files}] 完成 {Path(f.target).name}（{msg}）")
    return True, f"{total_files} 个文件就绪"


def installed_summary(models_root: Path, specs: list[ModelSpec]) -> dict[str, Any]:
    inst = [s for s in specs if model_status(s, models_root)["installed"]]
    return {
        "models_root": str(models_root),
        "total": len(specs),
        "installed": len(inst),
        "installed_ids": [s.id for s in inst],
        "have_mb": round(sum(model_status(s, models_root)["have_mb"] for s in inst), 1),
    }
