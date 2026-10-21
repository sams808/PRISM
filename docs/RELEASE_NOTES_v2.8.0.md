PRISM 2.8.0 — a Home page that configures the app, a Windows installer, the Raman block completed for Spectragryph users, database packs, and the fixes of a full audit.

This release also carries everything prepared for the untagged 2.7.0 (SAXS 2.0, see `RELEASE_NOTES_v2.7.0.md`), the cyclic-DTA / multi-rate kinetics rework and the XRD CIF-folder identification that landed on `main` after it.

## Home page and modules

- A **Home** page opens first. Every module is a card with an *Enabled* switch, a description and buttons to its workspaces. A fresh install starts with **every module off**; switch on what the computer is for and the choice is remembered per user. Disabled modules are hidden from the rail, not removed.
- **Presets** (Home page and Modules menu): *Raman station* (Raman, Fitting, Processing, Figures — the configuration for the spectrometer's computer), *All modules*, *None*.
- **Reference databases** on the Home page: what is installed (RRUFF spectra, AMCSD structures, XRD card databases, CIF folders, your Raman reference library), **Export databases…** writes them as one pack folder, **Import a database pack…** installs such a folder on another computer, where PRISM reads it directly. Share packs only with people holding the same database rights.
- Data files dropped anywhere on the window are imported; **File › Paste data from clipboard** imports a table copied from Excel, Origin or a text editor.

## Windows installer

- `build_exe.bat` now produces `dist/PRISM-<version>-setup.exe` (Inno Setup; per-user install, Start-menu entry, optional desktop icon, uninstaller, updates in place) and the portable `dist/PRISM-<version>-win64.zip`.
- `PRISM.exe --distcheck` builds the whole application off-screen, opens every workspace and exits 0, writing `prism_distcheck.log`; the build script runs it before packaging. The frozen build bundles numpy's submodules and xraydb's data table (the two things a PyInstaller build of this stack silently loses).

## Raman: what Spectragryph did, now in PRISM

- **Peak labels** and a **Peak table** (position, height, FWHM at half prominence, area, prominence; copy as text or save as CSV) in the Raman workspace; threshold as a percentage of the intensity range.
- **Cursor readout**: the value of every plotted spectrum at the mouse position.
- Line width, lines / points display, legend inside / outside / none; **Copy plot** (image) and **Copy data** (tab-separated) to the clipboard.
- **Raman ID — spectrum similarity**: rank RRUFF's measured spectra and your own reference library by the Pearson correlation (or a normalised Euclidean distance) of the whole spectrum, raw, smoothed or as first / second derivatives, optionally inside an x range. Hits show a score (1 = identical shape) and their curve.
- **Your Raman reference library**: add the query spectrum with name, laser wavelength, source and notes; rename, remove, export, import; included in database packs.
- **Calculations** gains: normalise to a value at x₀, subtract a range mean (simple baseline), remove a peak, SNV, MSC, median of spectra, derivatives to the 4th order, x-axis unit conversion (wavelength ⇄ wavenumber ⇄ Raman shift ⇄ photon energy, optional Jacobian) and x-axis calibration against reference lines (polynomial, residuals reported).
- The Raman module guide lists what was ported and what was not (Spectragryph's own file formats, acquisition, scripted sequences).

## Fixes (2026-10-21 audit)

- **Import**: a decimal-comma export without a recognised header (`100,0000 1000,0000`) imported as x = 100, y = 0. Latin-1 or UTF-16 files with a Raman / XRD header, and files with `#` banner lines before the header, failed with "Could not read table with flexible settings".
- **True Voigt width**: the width field is now the profile's actual half-width at every η (Olivero–Longbothum component widths); at η = 0.5 the profile was 18 % narrower than its field value.
- **HT-XRD peak tracking**: the table, plot and CSV said FWHM but carried the fit's half-width (values doubled); a peak drifting more than 1.5 HWHM per pattern was reported absent — the seed now climbs uphill from the previous centre.
- **Moving average** (Calculations, Raman smoothing, XAS μ smoothing): the first and last half-window points were pulled towards zero (a flat 100 came out at 54.5 at the ends).
- **Open project** keeps nothing of the previous session: the HT-XRD series of a project without one is cleared, and the Library's undo stack no longer resurrects spectra of the replaced project.
- `xraydb` is pinned in `requirements.txt`: the app did not start on an install of the two core requirement files (the SAXS core imports it).
- CI lint (ruff F/E9) green again; two Python scripts shipped as "XANES examples" removed from EXAMPLES.

## Verified independently

- CIF powder patterns against pymatgen's XRDCalculator on CaF₂, monoclinic ZrO₂ and α-Bi₂O₃: positions identical, intensities within a few percent once pymatgen's unresolved pairs are summed as PRISM does.
- Peak fitting, baselines, Tg methods, Guinier / Porod / Percus–Yevick limits, Bragg angle-to-energy, optical basicity, derivatives and integrals, LCF, Arrhenius, JCAMP-DX AFFN and SQZ/DIF/DUP, HT-XRD tracking: closed-form checks in the audit script, all passing.
