PRISM 2.5.0 — the databases-are-yours release: startup in seconds, an XRD database manager, and QualX-style phase-ID sessions.

## Startup
- **Launch time fixed** (reported ~2 minutes after a fresh pull): the app no longer imports PyTorch (GlassNet availability check) or Larch at startup — both load on first use, in a background thread. The splash appears within ~1 s and the heavy imports happen behind it. Warm start ~4 s.

## XRD ID — bring your own databases
- **PRISM ships no XRD reference data.** Register any QualX-format `.sq` you have the rights to use with **Add database…** / **Add folder…**: QualX-format files are indexed once locally (minutes, in the background); PRISM-format files (e.g. shared by a colleague) are used in place. An existing local database is migrated into the registry automatically.
- **Probe several databases in one search** — check any number in the list; results are merged, re-ranked, and tagged with their database. **Identical cards carried by several databases are shown once** (same code + phase + space group + quality).
- **Card filters**: crystal-system and card-quality check-droplists (quality entries reflect what your databases actually contain), plus a space-group substring field.
- Source-tag checkboxes always reflect the enabled databases' own contents — nobody sees tags their files don't carry.

## Phase ID sessions (XRD ID and Raman ID)
- **Query peak bars** on the plot; peaks explained by an accepted phase turn **gray** instead of vanishing.
- **Accepted phases stay overlaid** (muted colors) across iterative searches, QualX-style.
- **Clear** button starts the session over (recorded identifications are untouched — Ctrl+Z in the Library undoes those).

## UX
- **Fixed: duplicated Help menus** — every module toggle used to add another Help menu to the menu bar.
- **Every workspace now shows a plot on arrival** (query pattern / spectrum / selected inputs) instead of an empty axes, with auto-selection of the first spectrum where nothing was selected; Update plot buttons added on the ID pages.
- Checkable list entries (e.g. the database list) now render their checkmarks correctly in dark mode.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
