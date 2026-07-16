@echo off
rem One-command portable-exe release: PyInstaller build with the exclusion
rem list this app needs (the shared Python environment contains torch,
rem transformers, llvmlite, PyQt5 and other heavyweights that would
rem otherwise be silently bundled - the first unfiltered build was 883 MB
rem vs 349 MB with these exclusions), then zip dist\Dataapp into a single
rem shareable archive.
rem
rem Note: the exe excludes xraylarch; Larch-dependent XAS steps
rem (normalization/EXAFS) need a Python install + Dataapp.bat instead.
cd /d "%~dp0"

py -3.11 -m PyInstaller --noconfirm --clean --windowed --name Dataapp ^
  --exclude-module larch --exclude-module wx --exclude-module tkinter ^
  --exclude-module PyQt5 --exclude-module PyQt6 ^
  --exclude-module IPython --exclude-module jupyter --exclude-module nbformat ^
  --exclude-module notebook --exclude-module zmq ^
  --exclude-module torch --exclude-module transformers --exclude-module tokenizers ^
  --exclude-module llvmlite --exclude-module numba ^
  --exclude-module botocore --exclude-module boto3 ^
  --exclude-module h5py --exclude-module lxml ^
  --exclude-module cryptography --exclude-module paramiko ^
  qt_main.py
if errorlevel 1 (
  echo Build failed.
  pause
  exit /b 1
)

echo Zipping dist\Dataapp ...
powershell -NoProfile -Command "Compress-Archive -Path 'dist\Dataapp' -DestinationPath 'dist\Dataapp-portable.zip' -Force"
echo Done: dist\Dataapp-portable.zip
pause
