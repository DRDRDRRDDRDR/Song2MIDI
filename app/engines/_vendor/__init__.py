"""从 spotify/basic-pitch 0.4.0 发行包中取出的官方算法源码，仅做最小改动。

来源：https://pypi.org/project/basic-pitch/  版本 0.4.0
许可：Apache License 2.0（原始版权归 Spotify AB）

改动清单（仅此两处，不涉及算法逻辑）：
    1. 包内导入路径改写为相对导入（`from basic_pitch.constants` → `from .constants`）。
    2. mir_eval 与 resampy 改为可选导入。二者仅被 sonify()（把结果合成回音频试听）
       使用，不参与 model_output_to_notes / output_to_notes_polyphonic 的音符提取，
       做成可选可避免为一条用不到的代码路径拉入 numba 编译链。

保留原样不动的原因：音符提取的阈值处理、melodia 后处理、音高弯曲换算
存在大量细节经验，自行重写极易引入难以察觉的偏差。取用官方实现是更可靠的选择。
"""

__all__ = ["note_creation", "constants"]
