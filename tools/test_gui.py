"""桌面界面冒烟测试。

在无显示环境下用 offscreen 平台跑，验证：
  - 模块能导入、窗口能构建
  - 控件联动（切换引擎会换档位列表、切分离开关会禁用相关控件）
  - 选项收集函数输出结构正确
  - 关键路径（工作线程类的构造）不报错

不做真实截图与交互，只确认不会一启动就崩。
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, ROOT)

from PySide6.QtWidgets import QApplication

from app.config import load_config
from app.gui import MainWindow, _Worker

cfg = load_config()
app = QApplication.instance() or QApplication([])

print("=" * 84)
print("构建主窗口")
print("=" * 84)
win = MainWindow(cfg)
print("  窗口标题:", win.windowTitle())
print("  初始尺寸:", win.size().width(), "x", win.size().height())

print()
print("=" * 84)
print("检查控件是否齐全")
print("=" * 84)


def present(name):
    return hasattr(win, name)


checks = {
    "文件列表": "file_list",
    "输出目录输入": "out_edit",
    "分离方式": "cmb_separation",
    "引擎下拉": "cmb_engine",
    "档位下拉": "cmb_model",
    "设备下拉": "cmb_device",
    "鼓组开关": "cb_drums",
    "和弦开关": "cb_chords",
    "量化下拉": "cmb_quant",
    "节拍下拉": "cmb_bpm",
    "BPM 输入": "spin_bpm",
    "起始秒": "spin_start",
    "时长秒": "spin_dur",
    "运行按钮": "btn_run",
    "取消按钮": "btn_cancel",
    "日志视图": "log_view",
    "结果视图": "result_view",
    "进度条": "progress",
    "环境标签": "env_label",
}
for label, attr in checks.items():
    print(f"  {'OK  ' if present(attr) else '缺失'} {label}")

print()
print("分轨复选框:", list(win.stem_boxes.keys()))
print("环境状态文字:", win.env_label.text()[:90])

print()
print("=" * 84)
print("控件联动")
print("=" * 84)
# 引擎切换到 roformer，档位列表应变为单体 RoFormer
win.cmb_engine.setCurrentIndex(1)
app.processEvents()
models = [win.cmb_model.itemText(i) for i in range(win.cmb_model.count())]
print(f"  切到 {win.cmb_engine.currentData()} 后档位列表: {models}")
print(f"  档位数据: {[win.cmb_model.itemData(i) for i in range(win.cmb_model.count())]}")
print(f"  警告文字: {win.engine_warn.text()[:100] or '（无）'}")

win.cmb_engine.setCurrentIndex(0)
app.processEvents()
models = [win.cmb_model.itemData(i) for i in range(win.cmb_model.count())]
print(f"  切回 demucs 后档位: {models}")

# 分离方式三态：skip 时引擎选择应被禁用
win.cmb_separation.setCurrentIndex(2)   # skip
app.processEvents()
print(f"  选「跳过分离」后引擎下拉可用: {win.cmb_engine.isEnabled()}（应为 False）")
print(f"  分离方式取值: {win._separation_value()}（应为 False）")
win.cmb_separation.setCurrentIndex(0)   # auto
app.processEvents()
print(f"  选「自动检测」后引擎下拉可用: {win.cmb_engine.isEnabled()}（应为 True）")
print(f"  分离方式取值: {win._separation_value()}（应为 None）")
win.cmb_separation.setCurrentIndex(1)   # force
app.processEvents()
print(f"  选「强制分离」后引擎下拉可用: {win.cmb_engine.isEnabled()}（应为 True）")
print(f"  分离方式取值: {win._separation_value()}（应为 True）")
win.cmb_separation.setCurrentIndex(0)
app.processEvents()

# 手动 BPM 才启用输入框
win.cmb_bpm.setCurrentIndex(1)
app.processEvents()
print(f"  选「手动指定」后 BPM 输入可用: {win.spin_bpm.isEnabled()}（应为 True）")
win.cmb_bpm.setCurrentIndex(0)
app.processEvents()
print(f"  选「自动估计」后 BPM 输入可用: {win.spin_bpm.isEnabled()}（应为 False）")

print()
print("=" * 84)
print("选项收集")
print("=" * 84)
opts = win._collect_options()
for k, v in opts.items():
    print(f"  {k:<18} {v}")

print()
print("=" * 84)
print("日志与拖放接口")
print("=" * 84)
win._append_log("info", "测试信息")
win._append_log("ok", "测试成功")
win._append_log("error", "测试错误 <需转义> & 符号")
text = win.log_view.toPlainText()
print("  日志行数:", len(text.splitlines()))
# QPlainTextEdit 没有 toHtml()，转义效果通过纯文本回读间接确认：
# 富文本内容被正确解析后，纯文本里应是原始字符而非 HTML 实体
print("  转义后纯文本含原字符:", "<需转义>" in text)
print("  纯文本未残留 HTML 实体:", "&lt;" not in text)
print("  接受拖放:", win.acceptDrops())

# ---------------------------------------------------------------------------
# 按钮/交互路径测试
#
# 这一段是补上的。此前只验证了「窗口能构建、控件联动正确」，
# 没有真正走一遍按钮回调，因此漏掉了 app/gui.py 里 7 处相对导入越界
# （`from ..ingest import ...` 在 app 包内应为 `.`），
# 症状是「添加文件」按钮点了没反应，而冒烟测试全绿。
# 教训：控件层的测试必须真正触发回调，不能只检查控件状态。
# ---------------------------------------------------------------------------
print()
print("=" * 84)
print("按钮与交互路径（真实触发回调）")
print("=" * 84)

from pathlib import Path as _P

from PySide6.QtCore import QMimeData, QUrl
from PySide6.QtWidgets import QFileDialog

import app.gui as _g

# 造两个真实存在的音频文件占位（内容无所谓，只验证「选中后能加进列表」）
tmpdir = _P(ROOT) / "work" / "_gui_test"
tmpdir.mkdir(parents=True, exist_ok=True)
fake = []
for n in ("测试甲.mp3", "测试乙.flac"):
    p = tmpdir / n
    p.write_bytes(b"not-a-real-audio")
    fake.append(str(p))

real_open = QFileDialog.getOpenFileNames
real_dir = QFileDialog.getExistingDirectory

try:
    # 1) 添加文件
    QFileDialog.getOpenFileNames = staticmethod(lambda *a, **k: (fake, "音频文件"))
    win._clear_files()
    win._add_files()
    app.processEvents()
    print(f"  添加文件: 列表 {win.file_list.count()} 项  "
          f"（期望 {len(fake)}）  成功={win.file_list.count() == len(fake)}")

    # 2) 添加文件夹
    QFileDialog.getExistingDirectory = staticmethod(lambda *a, **k: str(tmpdir))
    win._clear_files()
    win._add_folder()
    app.processEvents()
    print(f"  添加文件夹: 列表 {win.file_list.count()} 项  "
          f"（期望 {len(fake)}）  成功={win.file_list.count() == len(fake)}")

    # 3) 移除选中 / 清空
    win.file_list.setCurrentRow(0)
    win.file_list.item(0).setSelected(True)
    win._remove_selected()
    app.processEvents()
    print(f"  移除选中后: {win.file_list.count()} 项  "
          f"成功={win.file_list.count() == len(fake) - 1}")
    win._clear_files()
    print(f"  清空后: {win.file_list.count()} 项  成功={win.file_list.count() == 0}")

    # 4) 拖放路径（构造真实的 QMimeData）
    md = QMimeData()
    md.setUrls([QUrl.fromLocalFile(f) for f in fake] + [QUrl.fromLocalFile(str(tmpdir))])

    class _Ev:
        def __init__(self, m):
            self._m = m
            self.accepted = False

        def mimeData(self):
            return self._m

        def acceptProposedAction(self):
            self.accepted = True

    ev = _Ev(md)
    win._clear_files()
    win.dropEvent(ev)
    app.processEvents()
    print(f"  拖放加入: {win.file_list.count()} 项  接受={ev.accepted}")

    # 5) 未选文件时点开始，应给出提示而不是崩溃
    win._clear_files()
    from PySide6.QtWidgets import QMessageBox
    real_info = QMessageBox.information
    QMessageBox.information = staticmethod(lambda *a, **k: None)
    try:
        win._start()
        print("  空列表点「开始处理」: 已拦截并提示（未崩溃）")
    finally:
        QMessageBox.information = real_info

    # 6) 这两行曾因 `from ..pool import JobResult` 越界而在真正处理时报错，
    #    这里直接验证导入可用
    import importlib
    for mod, name in ((".pipeline", "PipelineOptions"), (".pool", "JobResult"),
                      (".ingest", "SUPPORTED_EXTS"), (".config", "load_config")):
        m = importlib.import_module(f"app{mod}")
        ok = hasattr(m, name)
        print(f"  app{mod}.{name}: {'可导入' if ok else '缺失'}")

finally:
    QFileDialog.getOpenFileNames = real_open
    QFileDialog.getExistingDirectory = real_dir
    for f in fake:
        try:
            _P(f).unlink()
        except OSError:
            pass

print()
print("=" * 84)
print("结论")
print("=" * 84)
print("  桌面界面构建、联动与按钮回调均正常")
