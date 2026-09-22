@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
cd /d "%~dp0"

rem ---------------------------------------------------------------------------
rem  Song2MIDI 源码启动脚本
rem
rem  若已打包，直接双击 dist\Song2MIDI\Song2MIDI.exe 即可，无需本脚本。
rem
rem  Python 环境按以下优先级查找（先命中者胜出）：
rem    0. run.local.bat  —— 本机专用覆盖（已被 .gitignore 忽略，不会进仓库）
rem    1. 环境变量 SONGMIDI_PY / SONGMIDI_PYW
rem    2. 项目内的 .venv\Scripts\
rem    3. PATH 里的 pythonw / python
rem
rem  注意：PATH 里的 python 可能是没装依赖的基础解释器，
rem  所以前两项优先。若落到第 3 项后报缺模块，
rem  请在项目目录下建虚拟环境并装依赖（见下方提示）。
rem ---------------------------------------------------------------------------

rem 0) 本机专用覆盖
if exist "%~dp0run.local.bat" call "%~dp0run.local.bat"

set PY=
set PYW=

rem 1) 环境变量
if not defined PY  if defined SONGMIDI_PY  if exist "%SONGMIDI_PY%"  set "PY=%SONGMIDI_PY%"
if not defined PYW if defined SONGMIDI_PYW if exist "%SONGMIDI_PYW%" set "PYW=%SONGMIDI_PYW%"

rem 2) 项目内虚拟环境
if not defined PYW if exist ".venv\Scripts\pythonw.exe" set "PYW=.venv\Scripts\pythonw.exe"
if not defined PY  if exist ".venv\Scripts\python.exe"  set "PY=.venv\Scripts\python.exe"

rem 3) PATH
if not defined PYW for %%P in (pythonw.exe) do if not defined PYW set "PYW=%%~$PATH:P"
if not defined PY  for %%P in (python.exe)  do if not defined PY  set "PY=%%~$PATH:P"

set RUN=
if defined PYW set "RUN=%PYW%"
if not defined RUN if defined PY set "RUN=%PY%"

if not defined RUN (
    echo [错误] 未找到可用的 Python 环境。
    echo.
    echo 三种解决方式，任选其一：
    echo   1. 在项目目录下创建虚拟环境并安装依赖：
    echo        python -m venv .venv
    echo        .venv\Scripts\pip install -r requirements.txt -r requirements-heavy.txt
    echo   2. 设置环境变量 SONGMIDI_PY 指向你装好依赖的 python.exe
    echo   3. 若已打包，直接运行 dist\Song2MIDI\Song2MIDI.exe
    echo.
    pause
    exit /b 1
)

echo 使用解释器: %RUN%
rem 用 pythonw 启动，避免多弹一个控制台黑窗
start "" "%RUN%" "%~dp0main.py" %*
endlocal
