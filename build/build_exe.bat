@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
cd /d "%~dp0.."

set VENV_PY=C:\Users\DR\.workbuddy\binaries\python\envs\song2midi\Scripts\python.exe

echo ============================================================
echo  Song2MIDI 打包为 exe（onedir）
echo ============================================================
echo.

if not exist "%VENV_PY%" (
    echo [错误] 未找到虚拟环境：%VENV_PY%
    exit /b 1
)

echo [1/5] 检查 PyInstaller
"%VENV_PY%" -c "import PyInstaller" 2>nul
if errorlevel 1 (
    echo       未安装，正在安装...
    "%VENV_PY%" -m pip install -i https://mirrors.aliyun.com/pypi/simple/ pyinstaller
    if errorlevel 1 exit /b 1
)
"%VENV_PY%" -m PyInstaller --version

echo.
echo [2/5] 检查关键资源
for %%F in (
    "config.yaml"
    "models\nmp.onnx"
    "models\roformer\bs_roformer_4stems_ft\bs_roformer_4stems_ft.pth"
    "app\static\index.html"
) do (
    if exist "%%~F" (
        echo       存在  %%~F
    ) else (
        echo       [警告] 缺失  %%~F
    )
)

echo.
echo [3/5] 清理旧的构建产物
if exist "dist\Song2MIDI" rmdir /s /q "dist\Song2MIDI"
if exist "build\Song2MIDI" rmdir /s /q "build\Song2MIDI"

echo.
echo [4/5] 开始打包（体积约 3 GB，耗时可能 10-30 分钟，请耐心等待）
echo ------------------------------------------------------------
"%VENV_PY%" -m PyInstaller --noconfirm --clean --distpath dist --workpath build\work ^
    build\song2midi.spec
set RC=%errorlevel%
echo ------------------------------------------------------------
if not "%RC%"=="0" (
    echo [失败] PyInstaller 返回 %RC%
    echo        若为 ModuleNotFoundError，把缺失模块加入 build\song2midi.spec 的 hiddenimports
    pause
    exit /b %RC%
)

echo.
echo [5/5] 打包完成，统计体积
if exist "dist\Song2MIDI" (
    echo.
    echo  产物目录: %CD%\dist\Song2MIDI
    echo.
    for /f "usebackq" %%S in (`powershell -NoProfile -Command ^
        "'{0:N0}' -f ((Get-ChildItem -Recurse -File 'dist\Song2MIDI' | Measure-Object Length -Sum).Sum/1MB)"`) do (
        echo  总大小: %%S MB
    )
    echo  主程序: dist\Song2MIDI\Song2MIDI.exe
    echo.
    echo  提示: 把整个 dist\Song2MIDI 文件夹拷走即可使用，不能只拷 exe。
) else (
    echo [失败] 未生成 dist\Song2MIDI
)

echo.
pause
endlocal
