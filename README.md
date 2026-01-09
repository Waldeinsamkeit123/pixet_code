# pixet_code

This repository contains analysis and test code for PIXET-related studies.

## Requirements

The analysis code requires the following Python libraries:

- `numpy`
- `matplotlib`
- `uproot`
- `awkward`

They can be installed via `pip`:

```bash
pip install numpy matplotlib uproot awkward
# or conda:
conda install -c conda-forge numpy matplotlib uproot awkward
```


## Contents
- Scripts transfer .clog into .root files.
- Draw plots.
- Do analysis(remove Fluorescence)

## Notes
- Large binary files (e.g. `.root`, `.png`, `.clog`) are not tracked by git.
- Only representative test outputs are kept in the repository.

## Author
Han Wang