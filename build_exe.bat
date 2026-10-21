@echo off
rem One-command Windows release of PRISM:
rem   1. PyInstaller  -> dist\PRISM\PRISM.exe (+ _internal\)
rem   2. notes, licence and the no-Python database scripts next to the exe
rem   3. self-test     dist\PRISM\PRISM.exe --distcheck (must exit 0)
rem   4. Inno Setup   -> dist\PRISM-<version>-setup.exe
rem   5. zip          -> dist\PRISM-<version>-win64.zip (portable)
rem Run from anywhere; it works in the repository root. Pass --no-pause when
rem calling it from a script.
rem
rem The exclusion list matters: the shared Python environment contains
rem torch, transformers, llvmlite, PyQt5 and other heavyweights that would
rem otherwise be silently bundled (883 MB unfiltered vs ~350 MB). The exe
rem excludes xraylarch AND glasspy: Larch-dependent XAS steps and GlassNet
rem predictions need a Python install + PRISM.bat. glasspy is excluded as a
rem whole because a build that only excluded torch still bundled torch
rem (246 MB) through a transitive glasspy->lightning PyInstaller hook.
rem
rem --collect-submodules numpy: with numpy >= 2 the frozen app otherwise
rem dies on `numpy._core._exceptions` at startup (seen on another app of the
rem group). --collect-data xraydb ships xraydb's SQLite table, without which
rem every scattering factor, SAXS contrast and CIF powder pattern fails.
setlocal
cd /d "%~dp0"
set "NOPAUSE=%1"

set "VERFILE=%TEMP%\prism_build_version.txt"
py -3.11 -c "import sys; sys.path.insert(0, '.'); from core.qt_help import APP_VERSION; sys.stdout.write(APP_VERSION)" > "%VERFILE%" || goto :fail
set /p VER=<"%VERFILE%"
del "%VERFILE%" >nul 2>&1
if "%VER%"=="" (
  echo could not read APP_VERSION from core\qt_help.py
  goto :fail
)
echo [1/5] PyInstaller, PRISM %VER%
py -3.11 -m PyInstaller --noconfirm --clean --windowed --name PRISM ^
  --icon assets\prism.ico --add-data "assets;assets" --add-data "EXAMPLES;EXAMPLES" ^
  --collect-submodules numpy --collect-data xraydb --collect-submodules sqlalchemy --collect-submodules h5py ^
  --exclude-module larch --exclude-module glasspy ^
  --exclude-module wx --exclude-module tkinter ^
  --exclude-module PyQt5 --exclude-module PyQt6 ^
  --exclude-module IPython --exclude-module jupyter --exclude-module nbformat ^
  --exclude-module notebook --exclude-module zmq ^
  --exclude-module torch --exclude-module transformers --exclude-module tokenizers ^
  --exclude-module llvmlite --exclude-module numba ^
  --exclude-module botocore --exclude-module boto3 ^
  --exclude-module lxml ^
  --exclude-module cryptography --exclude-module paramiko ^
  --exclude-module pyarrow --exclude-module dask ^
  --exclude-module lightning --exclude-module pytorch_lightning ^
  --exclude-module pymatgen --exclude-module gemmi --exclude-module ase ^
  --exclude-module pytest --exclude-module pytestqt ^
  qt_main.py
if errorlevel 1 goto :fail

echo [2/5] notes, licence, database scripts
copy /y scripts\Download-RRUFF-database.bat dist\PRISM\ >nul
copy /y scripts\Download-RRUFF-database.ps1 dist\PRISM\ >nul
copy /y scripts\Download-AMCSD-structures.bat dist\PRISM\ >nul
copy /y LICENSE dist\PRISM\LICENSE.txt >nul
copy /y packaging\INSTALL.txt dist\PRISM\INSTALL.txt >nul

echo [3/5] self-test: dist\PRISM\PRISM.exe --distcheck
if exist dist\PRISM\prism_distcheck.log del dist\PRISM\prism_distcheck.log
dist\PRISM\PRISM.exe --distcheck
if errorlevel 1 (
  echo distcheck FAILED -- see dist\PRISM\prism_distcheck.log
  type dist\PRISM\prism_distcheck.log
  goto :fail
)
type dist\PRISM\prism_distcheck.log

echo [4/5] Inno Setup
set "ISCC=%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe"
if not exist "%ISCC%" set "ISCC=%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
if not exist "%ISCC%" (
  echo Inno Setup not found; install it with
  echo     winget install --id JRSoftware.InnoSetup -e --scope user
  goto :fail
)
"%ISCC%" /Q /DMyAppVersion=%VER% packaging\prism.iss
if errorlevel 1 goto :fail

echo [5/5] zip
powershell -NoProfile -Command "Compress-Archive -Path 'dist\PRISM' -DestinationPath 'dist\PRISM-%VER%-win64.zip' -CompressionLevel Optimal -Force"
if errorlevel 1 goto :fail

echo.
echo done:
dir /b dist\PRISM-%VER%-setup.exe dist\PRISM-%VER%-win64.zip
if not "%NOPAUSE%"=="--no-pause" pause
exit /b 0

:fail
echo Build failed.
if not "%NOPAUSE%"=="--no-pause" pause
exit /b 1
