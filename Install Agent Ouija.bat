@echo off
REM ─────────────────────────────────────────────────────────────────────────────
REM  Install Agent Ouija — double-click launcher for testers.
REM  Runs the same install.py the docs mention, but without asking the user to
REM  open a terminal. If Python 3.11+ isn't on PATH, the script prints a link
REM  and pauses so the window stays open long enough to read.
REM
REM  Flat GOTO structure on purpose — cmd's parenthesised if/else + nested
REM  %ERRORLEVEL% is a classic delayed-expansion trap that closed the window
REM  before hitting the pause on some testers' rigs.
REM ─────────────────────────────────────────────────────────────────────────────

title Install Agent Ouija
cd /d "%~dp0"

REM Switch this console to UTF-8 + tell Python to emit UTF-8 on
REM stdout/stderr so the emojis in install.py / uninstall.py
REM render on legacy cmd.exe rigs. Without this, a cp1252 console
REM raises UnicodeEncodeError on the first ✅ and the batch bails.
chcp 65001 >nul 2>&1
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"

echo.
echo ============================================================
echo   Agent Ouija Installer
echo ============================================================
echo   Working folder: %CD%
echo.

set "PYCMD="

REM Prefer the Windows py launcher (handles 3.11 / 3.12 / 3.13 cleanly).
where py >nul 2>&1
if not errorlevel 1 (
    set "PYCMD=py -3"
    goto :haspython
)

REM Fall back to plain "python" on PATH.
where python >nul 2>&1
if not errorlevel 1 (
    set "PYCMD=python"
    goto :haspython
)

echo   ERROR: no Python found on PATH.
echo.
echo   Install Python 3.11 or newer from https://www.python.org/downloads/
echo   (tick "Add python.exe to PATH" on the first installer screen).
echo   Then re-run this file.
echo.
pause
exit /b 1

:haspython
echo   Using: %PYCMD%
echo.

%PYCMD% install.py %*
set "RC=%ERRORLEVEL%"

echo.
if "%RC%"=="0" (
    echo   Installer finished cleanly. This window can stay open or close it.
) else (
    echo   Installer exited with code %RC%. Read the messages above to see why.
)
echo.
pause
exit /b %RC%
