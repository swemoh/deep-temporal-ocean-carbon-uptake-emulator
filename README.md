# Deep Temporal Emulator for Ocean Carbon Uptake

This project trains a temporal deep-learning emulator for monthly ocean carbon uptake diagnostics using gridded NetCDF model output. The codebase follows a preprocessing → windowing → training workflow:

1. Read and merge ocean-model variables from NetCDF files
2. Apply land masks and grid metadata
3. Build per-location feature/target dictionaries
4. Convert time series into 6-month lookback windows
5. Train a heteroscedastic LSTM or TCN model with uncertainty estimates

## Setup

```bash
conda create -n carbon-env python=3.12 -y
conda activate carbon-env
pip install -r requirements.txt
```

Dependencies include `pandas`, `numpy`, `xarray`, `joblib`, `torch`, `scikit-learn`, `matplotlib`, `seaborn`, and `jupyter`.

## Data requirements

This repository expects external ocean-model data and a mesh mask file. The preprocessing script reads monthly NetCDF fields such as SST, salinity, mixed-layer depth, heat and freshwater fluxes, stress/current components, ice fraction, and carbon flux variables. These data are not included in the repo and must be configured locally.

## Project structure

- `extraction/read_netcdf_files.py` — NetCDF ingestion and feature assembly
- `utils/build_data_dict.py` — temporal window construction
- `models/lstm.py` — attention-based heteroscedastic LSTM
- `models/tcn.py` — attention-based heteroscedastic TCN
- `notebooks/Uncertainty_training.ipynb` — training and calibration workflow
- `notebooks/Output_analysis_plots.ipynb` — evaluation and plotting

## Training workflow

Open `notebooks/Uncertainty_training.ipynb` and run the cells in order:

1. Set the local data path and output directory
2. Load the saved feature/target dictionaries
3. Build sliding-window tensors with `build_xy_from_dicts(...)`
4. Standardize the features and create PyTorch tensors
5. Train the LSTM or TCN model using a heteroscedastic Gaussian loss
6. Calibrate uncertainty on the validation/calibration split
7. Save the best checkpoint and metrics

## Notes

- The code contains placeholders such as `PATH`, `EXP-NAME`, and `DATA-PATH`; replace them with your local dataset locations before running.
- The model uses a 6-month lookback window and monthly time resolution.
- Data access is project-specific; contact the dataset owner if the raw simulation files are not available locally.