"""Song2MIDI —— 歌曲文件转多轨 MIDI。

模块划分：
    ffmpeg_tools  外部依赖定位（复用本机 ffmpeg，不安装、不改 PATH）
    config        配置加载
    ingest        任意音频格式 → 标准 wav
    engines       各转录引擎（音源分离 / 音高转录 / 鼓组 / 和弦）
    postprocess   音符量化与清理
    export        多轨 MIDI 写出
    pool          多进程并行调度
    pipeline      端到端编排
    api           本地 Web 服务
"""

__version__ = "0.1.0"
