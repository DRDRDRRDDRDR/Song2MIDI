# Song2MIDI

把歌曲文件转成多轨 MIDI。**桌面程序，全程本地处理，音频不出本机。**

- **输入**：mp3 / wav / flac / m4a / ogg / opus / aac / wma / aiff，含 24bit 无损
- **输出**：四轨分离后的多轨 MIDI（人声 / 鼓 / 贝斯 / 其他）+ 和弦轨，附音符 JSON
- **形态**：PySide6 桌面界面 / 命令行 / 打包后的 exe（三种入口，同一套内核）

```
音频文件
  │
  ├─ ffmpeg 解码 ────────── 统一为 44.1kHz 立体声（复用本机已有 ffmpeg）
  │
  ├─ 音源分离（二选一）
  │   ├─ Demucs htdemucs   GPU 约 16 秒/首   CPU 约 2.7 分钟/首
  │   └─ BS-RoFormer 4轨   GPU 约 2 分钟/首   CPU 约 79 分钟/首
  │
  ├─ 分轨专用转录
  │   ├─ vocals / bass / other   Basic Pitch ONNX（按分轨限复音数）
  │   ├─ drums                   自研分频段频谱通量 + 鼓件分类
  │   └─ chords                  CQT chroma + 和弦模板匹配
  │
  ├─ 后处理 ────────────── 节拍估计与量化、碎片合并、音域折叠、力度归一
  │
  └─ 导出 ──────────────── 多轨 .mid + 逐轨 .mid + 音符 .json
```

---

## 快速开始

### 方式一：exe（无需 Python 环境）

双击 `dist\Song2MIDI\Song2MIDI.exe`。

**注意：必须整个 `Song2MIDI` 文件夹一起拷走，不能只拷 exe** —— 依赖与模型都在 `_internal` 里。

把音频文件（或整个文件夹）拖到 `Song2MIDI.exe` 图标上，程序会**打开界面、预填文件并自动开始**。
之所以不直接静默批处理：exe 是无控制台的 GUI 程序，静默跑完用户看不到任何进度与结果，
体验上像是「双击了没反应」。

**界面会在启动后自动做一次环境自检**（后台进行，不阻塞操作）：状态栏与运行日志里
会给出「N 项阻塞 / N 项警告 / N 项正常」的结论，以及 torch 版本与 GPU 型号。
放在后台是必要的 —— 自检里首次加载 torch 约需 6 秒，放主线程会让窗口迟迟不出现。

### 方式二：从源码启动

双击 `run.bat`，或：

```bash
python main.py                # 桌面界面
python main.py --web          # 浏览器界面（备用入口）
python main.py --check        # 环境自检（命令行）
```

`run.bat` 按以下顺序查找 Python 环境，先命中者胜出：

| 顺序 | 来源 | 说明 |
|---|---|---|
| 0 | `run.local.bat` | 本机专用覆盖，已被 `.gitignore` 忽略，不会进仓库 |
| 1 | 环境变量 `SONGMIDI_PY` / `SONGMIDI_PYW` | 指向你装好依赖的解释器 |
| 2 | 项目内 `.venv\Scripts\` | 推荐做法 |
| 3 | PATH 里的 `pythonw` / `python` | 兜底。注意 PATH 里的解释器**可能没装本项目依赖** |

没有 `run.local.bat` 时，在项目目录下建虚拟环境即可：

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt -r requirements-heavy.txt
```

### 方式三：命令行批处理

```bash
# 先取 30 秒片段试跑，确认效果再处理整首
python main.py --cli "歌.flac" --start 40 --duration 30

# 批量处理整个目录，3 个文件并行
python main.py --cli "D:\Music" --jobs 3

# 用 BS-RoFormer 引擎，走 GPU
python main.py --cli "歌.flac" --engine roformer --device cuda

# 跳过分离，整首当一轨转录（快很多，准确度低）
python main.py --cli "歌.flac" --no-separate
```

---

## 独奏曲目与自动检测

四轨分离是为流行乐（人声/鼓/贝斯/其他）设计的。**纯钢琴曲、吉他独奏、弦乐独奏
没有这些成分，强行分离会拆出大量假鼓、割裂旋律。**

程序会**自动检测**：分析频段能量分布，若「无鼓（6k–16kHz 占比 < 2%）
且无贝斯（20–80Hz 占比 < 3%）」就自动跳过分离、整首直转。

实测一首 B 站提取的钢琴曲：

| 路径 | 结果 |
|---|---|
| 四轨分离（旧默认） | 490 个假鼓音符，drums 轨 112 个同音重叠（物理不可能） |
| 整首直转 | 252 音符，音域 A1–E6 完整覆盖 |

界面「分离方式」三档可手动覆盖：**自动检测（默认）/ 强制四轨分离 / 跳过分离**。
CLI 对应 `--separation auto|force|skip`（`--no-separate` 等价于 `skip`）。
检测阈值在 `config.yaml` 的 `solo_detect` 段，默认设得保守，宁可多做一次分离也不漏判。

---

## 模型管理（模型不随软件打包）

**软件本体只有几十 MB，模型按需下载。** 首次运行请在界面里点「模型浏览器」，
勾选需要的模型下载即可。

### 为什么外置

三个模型合计约 750 MB，而 CUDA 版 torch 本身已有 2.5 GB —— 打在一起会让产物
膨胀到 **5 GB**，超过 GitHub Release 的 **2 GB 单文件上限**，也没法常规分发。
外置之后：软件本体变小、模型可跨版本复用、用户只下自己需要的。

### 模型清单从哪来

清单托管在本仓库根目录的 [`models.json`](models.json)。软件点「刷新」即拉取它，
**改这个 JSON 就能让所有已安装的软件看到新模型，无需重新发版**。

拉取镜像按实测速度排序（`raw.githubusercontent.com` 在国内直连要 13 秒、
走代理直接超时，所以**不作首选**）：

| 镜像 | 直连 | 走代理 |
|---|---|---|
| `gh-proxy.com` | 663 ms | **284 ms** ← 首选 |
| `cdn.jsdelivr.net` | **637 ms** | 1716 ms |
| `raw.githubusercontent.com` | 13083 ms | 超时 ← 兜底 |

全部不可达时回退到内置清单（`app/models_builtin.json`），保证界面不空着。

### 模型存放位置

```
%LOCALAPPDATA%\Song2MIDI\models
```

选这里的理由：符合 Windows 惯例、不需要管理员权限、多个版本共享同一份。
可用环境变量 `SONG2MIDI_MODELS` 改到别的盘：

```bat
set SONG2MIDI_MODELS=D:\AIModels\Song2MIDI
```

**兼容旧版**：如果 exe 同级或项目根存在非空的 `models/`，会优先使用它 ——
从旧版本升级不必重下 750 MB。

### 可选模型一览

| 模型 | 大小 | 说明 |
|---|---|---|
| **Basic Pitch**（必需） | 0.2 MB | 通用多音转录。随包分发，开箱即用 |
| **Demucs htdemucs**（推荐） | 80 MB | 默认分离档位，速度质量平衡 |
| **钢琴专用转录**（推荐） | 164 MB | ByteDance 高分辨率钢琴转录，**含延音踏板** |
| BS-RoFormer 4stems | 503 MB | 分离质量最好，但 CPU 上 0.05x 实时，需 GPU |
| Demucs htdemucs_ft | 321 MB | 4 子模型集成，质量优于默认档，约 4 倍耗时 |
| Demucs hdemucs_mmi | 160 MB | 速度质量居中 |
| Demucs htdemucs_6s | 52 MB | 额外分出钢琴/吉他两轨（本项目暂未使用） |

**最小可用组合 = 244 MB**（Basic Pitch + htdemucs + 钢琴专用）。

---

## 分离引擎怎么选

| 引擎 | 设备 | 4 分钟歌耗时 | 质量 |
|---|---|---|---|
| **htdemucs** | GPU | **约 16 秒** | 基准 |
| htdemucs | CPU | 约 2.7 分钟 | 基准 |
| htdemucs_ft | CPU | 约 11 分钟 | 略优于 htdemucs |
| **BS-RoFormer** | GPU | **约 2 分钟** | 贝斯轨明显更纯 |
| BS-RoFormer | CPU | **约 79 分钟** | 同上，但慢到不实用 |

实测对比（同一 20 秒片段）：

| 指标 | htdemucs | BS-RoFormer | 判定 |
|---|---|---|---|
| **贝斯轨 20–100Hz 能量占比**（应高） | 0.393 | **0.733** | **RoFormer 更纯** |
| 人声轨低频占比（应低） | 0.0029 | 0.0041 | htdemucs 略好 |
| 分轨间平均相关性（越低越干净） | 0.0840 | 0.0819 | 基本持平 |
| 重构自洽性（误差越低越好） | **0.0381** | 0.1001 | htdemucs 好 2.6 倍 |

**结论：BS-RoFormer 的优势集中在贝斯轨，其余维度与 htdemucs 持平甚至略逊，
而代价是慢 8 倍（GPU 下）。** 日常用 htdemucs，需要更干净的贝斯时切 RoFormer。

**BS-RoFormer 在 CPU 上实测 0.05x 实时**（官方推理流程含 4 倍重叠），
4 分钟的歌要 79 分钟，**必须用 GPU 才实用**。界面里选到该引擎且未检测到
CUDA 时会直接给出警告。

---

## 环境

### 已验证的运行环境

| 项目 | 实际使用 |
|---|---|
| 系统 | Windows 10 (19045) |
| Python | 3.13.14（使用项目内的独立虚拟环境） |
| GPU | RTX 4060 Laptop 8GB（CUDA 12.6，cuDNN 91002，实测 4.0 TFLOPS） |
| torch | **2.14.0+cu126**（CUDA 版） |
| 关键库 | librosa 1.0.0、onnxruntime 1.30.0、demucs 4.1.0、msst 0.1.0、PySide6 6.11.0 |
| ffmpeg | **复用本机已有**的 gyan.dev full build，未安装、未改 PATH |

### 启用 GPU（必要，否则 BS-RoFormer 不可用）

PyPI 上 Windows 版 torch 是 **CPU-only** 构建，所有 CUDA 依赖都被限定为
`platform_system == "Linux"`。必须从 PyTorch 官方索引装：

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126
```

约 2.5 GB。本机实测走 Clash 代理约 909 KB/s～3 MB/s，耗时约 17 分钟。
索引里有与本项目同版本的 `torch-2.14.0+cu126`，**不存在降级风险**。
装完用 `python main.py --check` 确认 CUDA 可用。

### 网络前提与代理

| 域名 | 用途 | 状态 |
|---|---|---|
| `dl.fbaipublicfiles.com` | Demucs 权重（80 MB） | 可达 |
| `huggingface.co` | BS-RoFormer 权重（503 MB） | **DNS 被污染为 127.0.0.1，必须走代理** |
| `download.pytorch.org` | CUDA 版 torch | 可达（直连慢，建议走代理） |
| `mirrors.aliyun.com` / `pypi.tuna.tsinghua.edu.cn` | 装依赖 | 可达 |

**程序会自己探测代理**，不依赖环境变量（exe 双击运行时没有那些变量）。
它会同时测「连通性」与「速度」，选最快的一条 —— 因为本机实测直连
`download.pytorch.org` 虽然可达，但速度只有 94 KB/s，而走 Clash 代理是
909 KB/s，**差近 10 倍**。候选端口在 `config.yaml` 的 `network.proxy_ports` 里配置。

---

## 配置

`config.yaml`。所有键都可省略，缺失项自动回落默认值。

```yaml
separator:
  engine: demucs          # demucs | roformer
  model: htdemucs         # htdemucs | htdemucs_ft | hdemucs_mmi | mdx_extra
  device: auto            # auto | cuda | cpu

postprocess:
  quantize: "1/16"        # 量化网格；节奏自由的曲目建议设 off
  bpm: auto               # 自动估计。估计不准时直接写数字，如 128

transcription:
  per_stem:
    bass:
      max_polyphony: 1    # 贝斯几乎总是单音，限死能显著去错音
```

---

## 从 GitHub 获取编译产物

不用自己装环境，[Actions](https://github.com/DRDRDRRDDRDR/Song2MIDI/actions) 里
已配置好自动构建。打 tag 会自动发布 Release：

```bash
git tag v1.0.0 && git push origin v1.0.0
```

也可以手动触发：Actions 页面 → Build Release → Run workflow。

### 两个变体

| 变体 | 大小 | 适用 |
|---|---|---|
| **cpu** | 约 0.8 GB，单文件 | 没有 NVIDIA 显卡时选它 |
| **cuda** | 约 4.3 GB，**7z 分卷** | 有 NVIDIA 显卡时选它 |

⚠️ CUDA 版是分卷压缩 —— 必须把 `.7z.001`、`.002`、`.003` **全部下载**，
再对 `.001` 解压（7-Zip 会自动串起来）。

解压出的 `Song2MIDI` 文件夹要**整体保留**，不能只拷 exe：依赖都在 `_internal/`。

### CI 里为什么不下载模型

模型已外置，构建时不需要它们 —— 这既让构建更快，也避免了 CI 里再走一遍
国内拉模型的麻烦。CI 只断言「包里 models/ 不超过 5 MB」，确保模型没被误打进去。

---

## 打包成 exe

```bash
build\build_exe.bat
```

产出 `dist\Song2MIDI\`，实测约 **4.83 GB**。体积构成：

| 项目 | 体积 |
|---|---|
| `_internal/torch`（CUDA 运行时 DLL） | 3886 MB |
| `_internal/models`（BS-RoFormer 503 MB + Demucs 80 MB + ONNX 0.2 MB） | 583 MB |
| `_internal/llvmlite` | 115 MB |
| `_internal/PySide6` | 72 MB |
| `_internal/scipy` | 51 MB |
| `_internal/onnxruntime` | 36 MB |
| `Song2MIDI.exe` | 56 MB |

其中 torch 的单个 DLL 就有：`torch_cuda.dll` 1011 MB、`cublasLt` 507 MB、
cuDNN 三个 DLL 合计 880 MB。**这些是 GPU 推理必需的，不能删** ——
删错会导致运行时崩溃且报错信息完全不提缺哪个 DLL。

**三个已踩过的坑（写在这里避免重犯）**：

1. **不要为了省体积排除 torch 的内部子包**。曾把 `torch._inductor`、
   `torch._dynamo`、`torch.distributed` 加进 excludes，以为「纯推理用不到编译栈」，
   结果 `import torch` 直接失败 —— 它们是被 torch **顶层导入**的，
   而体积只省下约 44 MB（相对 4.8 GB 毫无意义）。
2. **必须注册 `torch/lib` 为 DLL 搜索目录**（见 `build/rthook_torch_dll.py`）。
   Windows 加载扩展模块时不会自动搜索该目录，缺这一步会抛
   `WinError 126 找不到指定的模块`，且不提示缺的是哪个 DLL。
3. **必须收集各包的非 `.py` 数据文件**（见 spec 里的 `collect_data_files`）。
   `demucs/remote/files.txt` 记录预训练权重的下载地址清单，缺了它会报
   `FileNotFoundError`，而上层提示却写「首次运行需要下载约 80 MB 权重」——
   把排查方向引向网络。同类必须收集的还有 `librosa` 的 `intervals.msgpack`、
   `resampy` 的 `kaiser_*.npz`、`pretty_midi` 的 `sf2`。

**另外两个只在打包后才出现的问题**：

- **进程跑完不退出**。任务已完成、输出已打印、MIDI 已生成，进程却常驻
  （实测 150 秒仍在）。原因是 onnxruntime / numba 创建的非守护线程让
  CPython 的解释器退出流程阻塞。解法是在 `main.py` 的 `__main__` 里，
  冻结环境下任务结束即 `os._exit(code)`。
- **GBK 控制台编码**。Windows 中文控制台是 cp936，`✓`/`✗`/`▸` 都不在其中，
  打印到它们会抛 `UnicodeEncodeError` **并中断整个流程** ——
  实测分离与转录已全部成功，只在最后打印「✓ 校验」那一行崩掉。
  已在入口加 `sys.stdout.reconfigure(errors="replace")` 兜底，
  并把符号换成 GBK 可表示的写法。

**为何只用 onedir 不出 onefile**：onefile 每次启动都要把 4.8 GB 解压到临时目录，
启动会长达数分钟、磁盘占用翻倍，实际不可用。

若不需要 GPU，可把 torch 换回 CPU 版（`pip install torch --force-reinstall`），
打包体积会降到约 1.6 GB，此时 onefile 也可用。

---

## 实测能力边界

### 鼓组转录（合成信号，答案已知）

| 鼓件 | 召回 | 精确 |
|---|---|---|
| 军鼓 | 100% | 100% |
| 开镲 | 100% | 100% |
| 底鼓 | 87.5% | 100% |
| 闭镲 | 85.7% | 100% |

簇召回率 87.5%，簇精确率 100%，**鼓件集合完全正确率 84.4%**。

### 音高转录

- CPU 上实时率约 **22 倍**（60 秒音频 2.7 秒），转录不是瓶颈
- 鼓与贝斯单轨质量最好；**吉他 / 合成器 / 弦乐（other 轨）质量不可控**，
  这是模型能力边界
- 混音直接转录会产生大量低频假音符（实测某曲 121 个音符里 96 个落在
  G1–G#1 次低频区）—— 这正是必须做音源分离的原因

### 真实曲目实测

`Le Castle Vania - Infinite Ammo`（PAYDAY 2 配乐），取 40–70 秒，htdemucs：

```
30 秒音频，总耗时 18.9 秒（1.6 倍实时，CPU）
节拍 125.854 BPM    145 个鼓击打簇间隔中位数 0.12 秒
  = 该速度下的十六分音符，说明击打落在正确的节拍网格上
鼓件分布：闭镲 143 / 军鼓 86 / 底鼓 44 / 通鼓 28 / 开镲 2
MIDI 回读校验：5 音轨 / 649 音符
```

`揽佬SKAI ISYOURGOD - 八方来财`（中文说唱），取 30–55 秒，htdemucs：

```
节拍 122.857 BPM
贝斯音域 A#0–A#1（MIDI 22–34）= 808 次低音的典型位置，说明分离与转录都正确
MIDI 回读校验：5 音轨 / 425 音符
```

### 已知不足

1. **通鼓误报偏多**（30 秒内 28 个，真实曲目不太可能）。该规则与底鼓边界模糊，
   可抬高 `drums.th_low_tom_min` 抑制。
2. **军鼓数量可能偏高**（占击打簇 59%）。中频宽带的拍手、合成器短音易被判成军鼓。
3. **和弦是模板匹配**，转位与加音和弦会被归到最接近的三和弦。
4. **BS-RoFormer 的重构自洽性不如 htdemucs**（误差 0.10 vs 0.038），
   意味着它对混音的重构没那么严丝合缝，但这不等于听感更差。
5. **「一键完美还原原曲 MIDI」不存在**。实际交付目标是"可用度高的多轨骨架"。

---

## 过程中的关键修正（记录以备后人）

**其一：频谱通量必须按频段 bin 数量归一。**
20–150Hz 在 2048 点 FFT 下只有约 7 个 bin，6k–16kHz 有约 460 个。
按频段直接求和会让高频在数量级上碾压低频 —— 实测底鼓检出 0 个、召回仅 58.9%。
按 bin 数取均值后恢复正常。

**其二：合成测试通过 ≠ 真实素材可用。**
开/闭镲的「1/e 衰减时间」判据在合成信号上完美，真实曲目完全颠倒
（30 秒内开镲 135、闭镲 4、crash 67）。原因是真实音乐高频内容本就连续，
测量窗口里几乎总有别的高频事件。最终改为「绝对下限 + 相对本曲中位数倍数」
双重判据才两边都成立。**此后任何音频算法改动都必须两边都验。**

**其三：节拍估计必须加感知先验。**
自相关在周期与 2 倍周期处往往同样强，直接取最大峰会整体错一倍。
实测某曲估出 63.7 BPM 而真实约 127 BPM —— 代价是量化网格差一倍
（1/16 在 63.7 BPM 下是 235ms，127 BPM 下是 118ms），所有音符时值被拉到错误格点。
改为对候选速度乘以 120 BPM 为中心的对数正态权重后修正。

**其四：代理选择必须实测速度，不能只判通不通。**
只测连通性会选出能连但极慢的路径（实测直连 pytorch.org 仅 94 KB/s，
走 Clash 909 KB/s）。此外本沙箱注入的 `HTTP_PROXY`（127.0.0.1:57910）
实测仅 6–80 KB/s，会静默拖慢所有下载 —— 这也是 `pip install PySide6`
曾卡死 11 分钟的原因（换清华源后 1.3 MB/s，一分钟装完）。

**其五：BS-RoFormer 有两套互不兼容的实现。**
lucidrains 的 `bs-roformer`（PyPI）与 ZFTurbo 的 `msst`（PyPI）虽然同名，
但架构代际不同。用前者加载 MSST 框架训练的 checkpoint，仅 32% 权重能对上
（mask estimator 张量尺寸差 2 倍）。必须用 `msst.models.bs_roformer`，
它可 100% 严格加载（1355/1355 张量）。

---

## 目录结构

```
Song2MIDI/
├─ main.py                 入口（默认桌面界面）
├─ run.bat                 双击启动（源码方式）
├─ config.yaml             配置
├─ app/
│  ├─ gui.py               PySide6 桌面界面
│  ├─ api.py               本地 Web 服务（备用）
│  ├─ ffmpeg_tools.py      外部依赖定位（复用本机 ffmpeg）
│  ├─ config.py            配置与路径（区分资源根 / 数据根）
│  ├─ net.py               代理探测与 HF 下载（含测速选优）
│  ├─ ingest.py            任意格式 → 标准 wav
│  ├─ notes.py             统一音符数据结构
│  ├─ pool.py              并行调度
│  ├─ postprocess.py       节拍估计、量化、清理
│  ├─ export.py            多轨 MIDI 导出与回读校验
│  ├─ pipeline.py          端到端编排
│  └─ engines/
│     ├─ separate.py            Demucs
│     ├─ separate_roformer.py   BS-RoFormer（msst）
│     ├─ pitch_onnx.py          Basic Pitch ONNX
│     ├─ drums.py               鼓组转录
│     ├─ chords.py              和弦识别
│     └─ _vendor/               从 basic-pitch 取出的官方算法（Apache-2.0）
├─ build/                  打包配置与脚本
├─ tools/                  自检、验证与批处理
├─ models/                 模型缓存（随 exe 分发）
├─ out/                    最终 MIDI 与音符 JSON
└─ work/                   中间产物
```

---

## 诊断工具

```bash
python main.py --check                        # 环境自检（含 GPU / RoFormer 检查）
python tools/test_drums.py --debug            # 鼓组引擎，合成信号答案已知
python tools/test_gui.py                      # 界面冒烟测试
python tools/compare_separators.py            # 两个分离引擎的质量与耗时对比
python tools/roformer_gpu_bench.py            # RoFormer 速度基准
python tools/transcribe_probe.py "歌.flac" --start 40 --duration 60
python tools/fetch_wheels.py                  # 大文件下载（支持断点续传）
```

---

## 避坑记录：`pip install basic-pitch` 不可用

`basic-pitch` 0.4.0 的元数据在 **Windows + Python≥3.11** 下强制要求
`tensorflow<2.15.1`，而 TF 2.15 最高只支持 Python 3.11 —— pip 解析阶段即失败。
但其实际代码对 TF 是**可选依赖**。本项目改为从 wheel 取出内置的 `nmp.onnx`
与官方算法源码，自写 ONNX 推理封装，彻底绕开 TF。顺带把官方逐窗口 batch=1
的推理改为批量推理，并加了官方接口没有的「按分轨限制复音数」。

---

## 许可

本项目代码由 WorkBuddy 生成。

- `app/engines/_vendor/` 下的 `constants.py` 与 `note_creation.py` 取自
  [spotify/basic-pitch](https://pypi.org/project/basic-pitch/) 0.4.0，
  版权所有 Spotify AB，Apache License 2.0，仅做最小改动（相对导入、
  mir_eval 与 resampy 改为可选导入），未改动任何算法逻辑。
- BS-RoFormer 权重 [SYH99999/bs_roformer_4stems_ft](https://huggingface.co/SYH99999/bs_roformer_4stems_ft)：Apache-2.0。
- 推理框架 [msst](https://pypi.org/project/msst/)（ZFTurbo/Music-Source-Separation-Training）。
- Demucs：MIT License。
