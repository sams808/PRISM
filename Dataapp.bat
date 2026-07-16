@echo off
rem Double-click launcher for Dataapp.
rem Uses the Windows py launcher to pick Python 3.11 explicitly — plain
rem "python" on this machine can resolve to a bare 3.10 install without the
rem app's dependencies. Starts windowless (pyw); if that fails, falls back
rem to a visible console run so the error is readable instead of silently
rem vanishing.
cd /d "%~dp0"

where pyw >nul 2>nul
if errorlevel 1 goto :console

start "" pyw -3.11 qt_main.py
exit /b 0

:console
py -3.11 qt_main.py
if errorlevel 1 pause
