from pathlib import Path
import pandas as pd
from pandas.errors import EmptyDataError

ROOT = Path(r"C:\BKHN\Data Science")
VALIDATION_DIR = ROOT / "results_gpm_validation"
OUTPUT_CSV = ROOT / "stations_fill_strategy.csv"

MIN_WET_DAYS = 200
CORR_REGRESSION = 0.5
CORR_BIAS = 0.40
MAX_ABS_MBE = 20.0

records = []

for station_dir in sorted(VALIDATION_DIR.glob("*_validation")):
    station = station_dir.name.replace("_validation", "")
    metric_file = station_dir / f"{station}_seasonal_metrics.csv"

    if not metric_file.exists():
        records.append({
            "station": station,
            "strategy": "DO_NOT_FILL",
            "wet_corr": None, "wet_rmse": None, "wet_mbe": None, "wet_days": 0,
            "dry_corr": None, "dry_rmse": None, "dry_mbe": None, "dry_days": 0,
        })
        continue

    try:
        df = pd.read_csv(metric_file)
    except EmptyDataError:
        records.append({
            "station": station,
            "strategy": "DO_NOT_FILL",
            "wet_corr": None, "wet_rmse": None, "wet_mbe": None, "wet_days": 0,
            "dry_corr": None, "dry_rmse": None, "dry_mbe": None, "dry_days": 0,
        })
        continue

    if df.empty or "season" not in df.columns:
        records.append({
            "station": station,
            "strategy": "DO_NOT_FILL",
            "wet_corr": None, "wet_rmse": None, "wet_mbe": None, "wet_days": 0,
            "dry_corr": None, "dry_rmse": None, "dry_mbe": None, "dry_days": 0,
        })
        continue

    df["season"] = df["season"].str.lower()

    wet = df[df["season"] == "wet"]
    dry = df[df["season"] == "dry"]

    if wet.empty or dry.empty:
        records.append({
            "station": station,
            "strategy": "DO_NOT_FILL",
            "wet_corr": None, "wet_rmse": None, "wet_mbe": None, "wet_days": 0,
            "dry_corr": None, "dry_rmse": None, "dry_mbe": None, "dry_days": 0,
        })
        continue

    w = wet.iloc[0]
    d = dry.iloc[0]

    wet_days = w["n_days"]
    wet_corr = w["correlation"]
    wet_mbe  = w["mbe"]

    if wet_days >= MIN_WET_DAYS and wet_corr >= CORR_REGRESSION:
        strategy = "LINEAR_REGRESSION"
    elif (
        wet_days >= MIN_WET_DAYS and
        CORR_BIAS <= wet_corr < CORR_REGRESSION and
        abs(wet_mbe) <= MAX_ABS_MBE
    ):
        strategy = "FILL_PLUS_BIAS"
    else:
        strategy = "DO_NOT_FILL"

    records.append({
        "station": station,
        "strategy": strategy,
        "wet_corr": wet_corr,
        "wet_rmse": w["rmse"],
        "wet_mbe": wet_mbe,
        "wet_days": wet_days,
        "dry_corr": d["correlation"],
        "dry_rmse": d["rmse"],
        "dry_mbe": d["mbe"],
        "dry_days": d["n_days"],
    })

out = pd.DataFrame(records)

out = out.sort_values(
    ["strategy", "wet_corr"],
    ascending=[True, False]
)

out.to_csv(OUTPUT_CSV, index=False)

print("Saved fill strategy to:")
print(OUTPUT_CSV)

print("\nStrategy summary:")
print(out["strategy"].value_counts())

