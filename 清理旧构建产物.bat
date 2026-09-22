@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
cd /d "%~dp0"

rem ===========================================================================
rem  清理旧构建产物（可反复运行，幂等）
rem
rem  为什么单独做成脚本：这些目录合计约 15 GB，但常被杀软实时扫描或索引服务
rem  占用，导致程序化删除失败。等你空闲时双击本脚本即可。
rem
rem  本脚本只删下面这几个「确定没用」的东西，不碰 dist\ 与源码：
rem    _旧构建产物_可整目录删除\   历次打包的中间产物与旧版本
rem    dist_v7\                    旧版本（已被 dist\ 取代）
rem    build\work*\                PyInstaller 中间目录
rem
rem  若提示「另一个程序正在使用」，说明仍被占用，稍后再跑一次即可。
rem ===========================================================================

echo.
echo ============================================================
echo   清理旧构建产物
echo ============================================================
echo.

set /a FREED=0

rem ---- 1) 待删目录 ----
if exist "_旧构建产物_可整目录删除" (
    echo [1/3] 删除 _旧构建产物_可整目录删除 ...
    rmdir /s /q "_旧构建产物_可整目录删除" 2>nul
    if exist "_旧构建产物_可整目录删除" (
        echo       部分内容被占用，已尽力删除，稍后可再跑一次
    ) else (
        echo       完成
    )
) else (
    echo [1/3] _旧构建产物_可整目录删除 不存在，跳过
)

rem ---- 2) 旧版本 dist_v7 ----
if exist "dist_v7" (
    echo [2/3] 删除 dist_v7 ...
    rmdir /s /q "dist_v7" 2>nul
    if exist "dist_v7" (
        echo       被占用，未能删除（可能正开着其中的 exe）
    ) else (
        echo       完成
    )
) else (
    echo [2/3] dist_v7 不存在，跳过
)

rem ---- 3) 打包中间目录 ----
if exist "build" (
    echo [3/3] 删除 build\work* ...
    for /d %%D in ("build\work*") do (
        rmdir /s /q "%%D" 2>nul
    )
    echo       完成
) else (
    echo [3/3] build 不存在，跳过
)

echo.
echo ============================================================
echo   当前状态
echo ============================================================
echo.
for /d %%D in (*) do (
    if /i "%%D"=="dist" (
        echo   [保留] %%D   ^<-- 程序入口，请勿删除
    )
)
if exist "dist_v7" echo   [未删] dist_v7   ^<-- 仍被占用
if exist "_旧构建产物_可整目录删除" echo   [未删] _旧构建产物_可整目录删除   ^<-- 仍被占用

echo.
echo 程序入口: %~dp0dist\Song2MIDI\Song2MIDI.exe
echo.
echo 提示：必须整个 dist\Song2MIDI 文件夹一起使用，不能只拷 exe。
echo.
pause
