# phaseexp
Phase Harmonic Scattering for Audio Signals

## Respository dependency

This repository uses some functions defined in https://github.com/edouardoyallon/pyscatwave, start by cloneing it at the same level as this repository :
```
your_directory
  |-- phaseexp/
  |-- pyscatwave/
```

## Create a conda environment with used packages:

```
chmod +x setup_conda_env.sh
bash setup_conda_env.sh
```
You will need to confirm the installation of several packages during the process.

Then, activate the new environment:
```
source activate phc
```

## Experiments

### Staircase / Cantor / Lena experiments

`make_figs.py` runs experiments on multiple types of 1D signals such as staircase functions, Lena lines or Cantor sets. The choice of signal is made using the variable `signal_name` which can take values in `["lena", "cantor", "staircase"]`.
Other variables are used to define hyperparameters for the embedding.
If not available, you can turn off the use of GPU by setting `cuda = False`.

### Music Experiments

`main_invprob.py` runs experiments on music files following the path `../data/gen_phaseexp_inv/*.wav` from the code directory.
The sampling rate can be adjusted with the variable `new_sr`.
