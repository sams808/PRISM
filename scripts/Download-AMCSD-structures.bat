@echo off
rem Downloads the AMCSD crystal-structure database (rruff.net) -- needed
rem only for the Raman ID workspace's "Overlay candidate's XRD (CIF)"
rem button. No Python needed. Roughly 66 MB; a couple of minutes.
rem Progress is logged to amcsd_download.log next to this script.
rem
rem Citation: Downs & Hall-Wallace (2003), "The American Mineralogist
rem Crystal Structure Database." American Mineralogist 88, 247-250.
cd /d "%~dp0"

if not exist "PRISM.exe" (
  echo PRISM.exe not found next to this script. Run it from inside the
  echo portable PRISM folder ^(the one PRISM.exe itself is in^).
  pause
  exit /b 1
)

echo Downloading and indexing AMCSD structures...
PRISM.exe --build-amcsd-cache
if errorlevel 1 (
  echo.
  echo FAILED -- see amcsd_download.log for details.
  pause
  exit /b 1
)

echo.
echo Done.
pause
