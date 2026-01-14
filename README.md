# pixet_code

This repository contains analysis and test code for PIXET-related studies.

## Requirements

The analysis code requires the following Python libraries:

- `numpy`
- `matplotlib`
- `uproot`
- `awkward`

!!warning: if need to run `fit_energy.py`, you need another environment with `ROOT` package, you can follow: [Root install](https://root.cern.ch/install/). Because `ROOT` and `uproot` is conflict, you may need 2 different environment.

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
- `new_remove_flu__and_save_root.py` : Modified remove Fluorescence(in case more than 1 Fluorescence clusters in 1 event), draw new plots, and save root files. `analyze_remove_flu.py` pick the minimal cluster_energy<30 and cluster_ncells<=2 cluster as Fluorescence cluster, but `new_remove_flu__and_save_root.py` pick all  cluster_energy<30 and cluster_ncells<=2 clusters as Fluorescence clusters.
- `fit_energy.py` : Do fit using root. Need `ROOT` package

## Notes
- Large binary files (e.g. `.root`, `.png`, `.clog`) are not tracked by git.
- Only representative test outputs are kept in the repository.

## Author
Han Wang