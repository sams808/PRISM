# Downloads the RRUFF Raman reference database (rruff.net) and builds
# PRISM's local search index -- no Python needed, works from this portable
# folder (calls PRISM.exe itself, headlessly). Roughly 400-500 MB
# downloaded; takes several minutes. Progress is logged to
# rruff_download.log next to this script -- open it if this window looks
# stuck. Safe to re-run: an interrupted download resumes instead of
# starting over.
#
# Citation: Lafuente, Downs, Yang & Stone (2015), "The power of databases:
# the RRUFF project."
Set-Location -Path $PSScriptRoot

$exe = Join-Path $PSScriptRoot "PRISM.exe"
if (-not (Test-Path $exe)) {
    Write-Host "PRISM.exe not found next to this script. Run it from inside the portable PRISM folder." -ForegroundColor Red
    exit 1
}

Write-Host "Downloading and indexing the RRUFF database..."
Write-Host "(this can take several minutes -- see rruff_download.log for live progress)"
& $exe --build-rruff-cache
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "FAILED -- see rruff_download.log for details." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Done. Launch PRISM.exe and open the Raman ID workspace." -ForegroundColor Green
