from pathlib import Path
import pandas as pd
import numpy as np
import xarray as xr
from scipy.stats import pearsonr

# ==================================================
# CONFIG
# ==================================================
ROOT = Path(r"C:\BKHN\Data Science")

NOAA_DIR = ROOT / "NOAA"   # NOAA station CSV
GPM_DIR  = ROOT / "subset_GPM_3IMERGDE_2015_2024"
OUT_DIR  = ROOT / "results_gpm_validation"

OUT_DIR.mkdir(exist_ok=True)

# columns
DATE_COL = "DATE"
NOAA_COL = "PRCP"

# seasons (theo nghiệp vụ của anh)
WET_MONTHS = [10, 11, 12, 1, 2, 3, 4]
DRY_MONTHS = [5, 6, 7, 8, 9]

# GPM precipitation variable candidates
PPT_CANDIDATES = ["precipitationCal", "precipitation", "precip"]

# ==================================================
# METRIC FUNCTIONS
# ==================================================
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mbe(y_true, y_pred):
    return np.mean(y_pred - y_true)

# ==================================================
# READ GPM DAILY (1 FILE = 1 DAY)
# ==================================================
def read_gpm_daily(date, lat, lon):
    ymd = date.strftime("%Y%m%d")
    files = list(GPM_DIR.glob(f"*{ymd}*.nc4"))
    if not files:
        return np.nan

    f = files[0]

    try:
        with xr.open_dataset(f, mask_and_scale=True) as ds:
            var = next((v for v in PPT_CANDIDATES if v in ds.data_vars), None)
            if var is None:
                return np.nan

            da = ds[var]
            lat_name = next(c for c in da.coords if "lat" in c.lower())
            lon_name = next(c for c in da.coords if "lon" in c.lower())

            return float(
                da.sel(
                    {lat_name: lat, lon_name: lon},
                    method="nearest"
                ).values
            )
    except Exception:
        return np.nan

# ==================================================
# MAIN LOOP
# ==================================================
for noaa_file in sorted(NOAA_DIR.glob("*.csv")):
    station = noaa_file.stem
    print(f"\nProcessing {station}")

    station_dir = OUT_DIR / f"{station}_validation"
    station_dir.mkdir(exist_ok=True)

    # --------------------------
    # LOAD NOAA
    # --------------------------
    df = pd.read_csv(noaa_file)

    if not {"DATE", "LATITUDE", "LONGITUDE", "PRCP"}.issubset(df.columns):
        print("  -> Skip (missing columns)")
        continue

    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.sort_values("DATE")

    lat = float(df["LATITUDE"].iloc[0])
    lon = float(df["LONGITUDE"].iloc[0])

    # --------------------------
    # ALIGN NOAA – GPM (DAY BY DAY)
    # --------------------------
    gpm_vals = [
        read_gpm_daily(d, lat, lon)
        for d in df["DATE"]
    ]

    aligned = pd.DataFrame({
        "DATE": df["DATE"],
        "LATITUDE": lat,
        "LONGITUDE": lon,
        "PRCP": df["PRCP"],
        "GPM": gpm_vals,
    })

    aligned_path = station_dir / f"{station}_aligned.csv"
    aligned.to_csv(aligned_path, index=False)
    print(f"  -> Saved {aligned_path.name}")

    # --------------------------
    # SEASONAL METRICS
    # --------------------------
    aligned = aligned.dropna(subset=["PRCP", "GPM"])
    if aligned.empty:
        print("  -> No valid data for metrics")
        continue

    aligned["month"] = aligned["DATE"].dt.month

    records = []

    for season, months in {
        "WET": WET_MONTHS,
        "DRY": DRY_MONTHS
    }.items():

        sub = aligned[aligned["month"].isin(months)]
        if len(sub) < 30:
            continue

        y_true = sub["PRCP"].values
        y_pred = sub["GPM"].values

        corr, _ = pearsonr(y_true, y_pred)

        records.append({
            "station": station,
            "season": season,
            "n_days": len(sub),
            "correlation": corr,
            "rmse": rmse(y_true, y_pred),
            "mbe": mbe(y_true, y_pred),
        })

    metrics_df = pd.DataFrame(records)
    metrics_path = station_dir / f"{station}_seasonal_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    print(f"  -> Saved {metrics_path.name}")

print("\nDONE")
