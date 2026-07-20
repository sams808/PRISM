@echo off
rem Downloads the RRUFF Raman reference database (rruff.net) and builds
rem PRISM's local search index -- no Python needed, works from this
rem portable folder (calls PRISM.exe itself, headlessly). Roughly
rem 400-500 MB downloaded; takes several minutes. Progress is logged to
rem rruff_download.log next to this script -- open it in Notepad if this
rem window looks stuck. Safe to re-run: an interrupted download resumes
rem instead of starting over.
rem
rem Citation: Lafuente, Downs, Yang & Stone (2015), "The power of
rem databases: the RRUFF project."
cd /d "%~dp0"

if not exist "PRISM.exe" (
  echo PRISM.exe not found next to this script. Run it from inside the
  echo portable PRISM folder ^(the one PRISM.exe itself is in^).
  pause
  exit /b 1
)

echo Downloading and indexing the RRUFF database...
echo ^(this can take several minutes -- see rruff_download.log for live progress^)
PRISM.exe --build-rruff-cache
if errorlevel 1 (
  echo.
  echo FAILED -- see rruff_download.log for details.
  pause
  exit /b 1
)

echo.
echo Done. Launch PRISM.exe and open the Raman ID workspace.
pause
