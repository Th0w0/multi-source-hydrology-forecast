# Multi-source flood detection and Damage Analysis system

This project implements a data processing pipeline and an AI-based forecasting model using multi-source hydrological and meteorological datasets.  
The instructions below explain how to set up the environment, understand the project structure, and run the full pipeline.

---

## 1. Project Structure

```text
data/
  raw/          # original downloaded datasets (ERA5-Land, IMERG, CPC, GRDC...)
  cleaned/      # cleaned & aligned data (after basic QC)
  processed/    # fully preprocessed data ready for model input

src/
  preprocessing/   # all preprocessing steps (cleaning, merging, area-weighted, imputation, normalization, sequences)
  models/          # forecasting model (LSTM, trainer, metrics)
  evaluation/      # evaluation scripts (return periods, comparison, visualization)
  utils/           # helper utilities (config loader, geo functions, etc.)

scripts/           # shell scripts to run preprocessing, training, and evaluation

notebooks/         # optional Jupyter notebooks for exploration and visualization
reports/           # project report or final documents

## 2. Setup Environment
pip install -r requirements.txt


## 3. Run the Program
Step 1 — Preprocess Data

Convert raw datasets into cleaned → processed datasets.

bash scripts/preprocess.sh

Step 2 — Train the Model
bash scripts/train.sh

Step 3 — Evaluate the Results
bash scripts/evaluate.sh
