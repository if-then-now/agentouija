@echo off
REM ─────────────────────────────────────────────────────────────────────────────
REM  Uninstall Agent Ouija — double-click launcher for testers.
REM  Runs uninstall.py which itself asks for a level (data / keys / full) and
REM  requires a typed YES before deleting anything. This .bat just saves the
REM  tester from opening a terminal.
REM
REM  Flat GOTO structure on purpose — cmd's parenthesised if/else + nested
REM  %ERRORLEVEL% is a classic delayed-expansion trap that closed the window
REM  before hitting the pause on some testers' rigs.
REM ─────────────────────────────────────────────────────────────────────────────

title Uninstall Agent Ouija
cd /d "%~dp0"

REM Switch this console to UTF-8 + tell Python to emit UTF-8 on
REM stdout/stderr so the emojis in uninstall.py render on legacy
REM cmd.exe rigs. Without this, a cp1252 console raises
REM UnicodeEncodeError on the first ✅ and the batch bails.
chcp 65001 >nul 2>&1
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"

echo.
echo ============================================================
echo   Agent Ouija Uninstaller
echo ============================================================
echo   Working folder: %CD%
echo.
echo   The uninstaller asks for a level (data / keys / full) and
echo   requires you to type YES before it deletes anything. If
echo   you double-clicked this by accident, close the window or
echo   press Ctrl+C to abort.
echo.

set "PYCMD="

where py >nul 2>&1
if not errorlevel 1 (
    set "PYCMD=py -3"
    goto :haspython
)

where python >nul 2>&1
if not errorlevel 1 (
    set "PYCMD=python"
    goto :haspython
)

echo   ERROR: no Python found on PATH. Nothing to uninstall from
echo   the Python side; delete this folder manually if you want
echo   the repo gone.
echo.
pause
exit /b 1

:haspython
echo   Using: %PYCMD%
echo.

%PYCMD% uninstall.py %*
set "RC=%ERRORLEVEL%"

echo.
if "%RC%"=="0" (
    echo   Uninstall finished. Delete the folder %CD% manually to
    echo   finish removing Agent Ouija from this machine (PowerShell
    echo   can't delete the folder it's running from).
) else (
    echo   Uninstall exited with code %RC%. Read the messages above.
)
echo.
pause
exit /b %RC%
