# Multi-source Hydrology Forecast

This project implements a data processing pipeline for multi-source hydrological and meteorological datasets, supporting two main workflows: **Station-based** and **Grid-based**.

## 1. Data Setup

To run the pipeline, you must first download the required datasets and place them in the correct directory structure.

### Step 1: Download Data
Download the data archive from the following Google Drive link:
[Google Drive Link](https://drive.google.com/drive/folders/1vL67ak6tTVeQAzjSVoxWxam4rfcOHDAo?usp=drive_link)

### Step 2: Extract Data
Extract the downloaded file into the `data/` directory at the root of this project.

### Step 3: Verify Structure
Ensure your `data/` directory looks like this:

```text
data/
├── grdc/                     # GRDC Station Data
├── BasinATLAS_v10_lev08.csv  # HydroATLAS Data
├── HydroBasin.csv            # HydroBASINS CSVs
├── precipitation_grid/       # Grid Precipitation
├── era5.grib                 # ERA5 Data
├── era5_merged/              # Merged ERA5 Data
└── g/                        # TIGGE/GRIB Data (e.g., 2025_01.grib)
```

## 2. Environment Setup

Install the required Python dependencies:

```bash
pip install -r requirements.txt
```

## 3. Running the Pipeline

There are two main workflows available. You can run them using the provided shell scripts or directly via Python.

### Workflow 1: NOAA Station Data
This workflow processes station-based data (Step 1 -> Step 2 -> Step 3 -> Step 4 -> Step 5).

**Using Shell Script:**
```bash
sh run_station_workflow.sh
```

**Using Python:**
```bash
python src/run_pipeline.py station --start_date=2019-01-01 --end_date=2024-12-31
```

**Output:**
Results will be saved in `output/final_station/`.

---

### Workflow 2: NOAA Grid Data
This workflow processes grid-based data (Step 1 -> Step 2b -> Step 3b -> Step 4 -> Step 5).

**Using Shell Script:**
```bash
sh run_grid_workflow.sh
```

**Using Python:**
```bash
python src/run_pipeline.py grid --start_date=2000-01-01 --end_date=2025-12-31
```

**Output:**
Results will be saved in `output/final_grid/`.
