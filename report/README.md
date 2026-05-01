# Final Report

## Build

```bash
cd report
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or with `latexmk`:

```bash
latexmk -pdf main.tex
```

## Files

- `main.tex` — single-file 5-6 page report skeleton with TODO comments.
- `refs.bib` — bibliography (FD, SFD, randomized SVD).
- Figures pulled from `../figures/` via `\graphicspath`.
