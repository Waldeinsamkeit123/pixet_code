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
- `convert_clog_to_root.py` : transfer .clog into .root files.
- `draw_under40_plot.py` : Draw plots.
- `analyze_remove_flu.py` : Do analysis(remove Fluorescence), and draw new plots.

## Notes
- Large binary files (e.g. `.root`, `.png`, `.clog`) are not tracked by git.
- Only representative test outputs are kept in the repository.

## Author
Han Wang