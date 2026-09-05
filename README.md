# Deep Temporal Emulator for Ocean Carbon Uptake

## Abstract

The global ocean carbon sink is a critical component of the Earth’s climate, but current ocean models are limited in their predictive capabilities because of the high computational cost that a biogeochemical (BGC) model, required to simulate air-sea CO2 fluxes, entails. Our research investigates the reconstruction of ocean CO2 flux focusing on local physical oceanographic drivers, such as sea surface temperature, salinity, and sea-ice coverage, which are outputs of simulations conducted with a global ocean BGC model. We frame the flux estimation problem as a temporal sequence problem and evaluate the efficacy of both Long Short-Term Memory (LSTM) and Temporal Convolutional Networks (TCN) architectures augmented with attention mechanisms.

The models ingest eight different physical parameters over a six-month sliding window at each grid point without explicit spatial information, and are designed to identify and weigh the historical physical states that most strongly influence current ocean-atmosphere carbon fluxes. Our framework employs LSTMs to capture sequential dependencies through recurrent hidden states. Moreover, we evaluate TCNs as a distinct alternative that uses dilated convolutions to identify multi-temporal patterns. Besides, we build a probabilistic framework utilizing Gaussian Negative Log-Likelihood (NLL) loss to account for aleatoric uncertainty in the dynamical drivers of ocean CO2 flux. To ensure credible reconstructions, we apply Conformal Prediction as a post-hoc calibration layer. This methodology yields calibrated uncertainty intervals with guaranteed 95% coverage.

Our results show that for most oceanographic regimes deep temporal models can function as reliable digital twins for point-wise carbon flux estimates. The proposed framework provides a data-driven foundation for reconstruction of air-sea carbon fluxes through an AI emulator built upon dynamic physical parameters.

## Authors

Sweety Mohanty<sup>1,2</sup> ([ORCID](https://orcid.org/0009-0004-2733-290X)), Daniyal Kazempour<sup>1</sup>, Andrea Göhring<sup>1</sup>, Willi Rath<sup>2</sup>, Lavinia Patara<sup>2</sup>, and Peer Kröger<sup>1</sup>

<sup>1</sup> University of Kiel, Germany  
<sup>2</sup> GEOMAR Helmholtz Centre for Ocean Research Kiel, Germany

## Contact

- University of Kiel: `smo, dka, pkr @ informatik.uni-kiel.de`, `agoehring @ leibniz-kiel.de`
- GEOMAR Helmholtz Centre for Ocean Research Kiel: `wrath, lpatara @ geomar.de`

# Project

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