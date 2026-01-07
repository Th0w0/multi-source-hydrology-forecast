from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(r"C:\BKHN\Data Science\results_gpm_validation")

CSV_SUFFIX = "_aligned.csv"
FIG_NAME = "noaa_vs_nasa_timeseries_overlap.png"

DATE_COL = "DATE"

NOAA_CANDIDATES = ["PRCP_NOAA", "PRCP", "precip", "precip_noaa"]
NASA_CANDIDATES = ["PRCP_GPM", "GPM", "precip_gpm", "precipitationCal"]

INVALID_MIN = 0.0

for station_dir in sorted(ROOT.glob("*_validation")):
    if not station_dir.is_dir():
        continue

    csv_files = list(station_dir.glob(f"*{CSV_SUFFIX}"))
    if not csv_files:
        print(f"Skip {station_dir.name} (no aligned csv)")
        continue

    csv_path = csv_files[0]
    print(f"Processing {csv_path.name}")

    df = pd.read_csv(csv_path, parse_dates=[DATE_COL])

    noaa_col = next((c for c in NOAA_CANDIDATES if c in df.columns), None)
    nasa_col = next((c for c in NASA_CANDIDATES if c in df.columns), None)

    if noaa_col is None or nasa_col is None:
        print(f"  -> Skip (cannot detect NOAA/GPM columns)")
        print(f"     Columns found: {list(df.columns)}")
        continue

    df = df[[DATE_COL, noaa_col, nasa_col]].sort_values(DATE_COL)

    df = df[
        (df[noaa_col].notna()) &
        (df[nasa_col].notna()) &
        (df[noaa_col] >= INVALID_MIN) &
        (df[nasa_col] >= INVALID_MIN)
    ]

    if df.empty:
        print("  -> Skip (no overlap data)")
        continue

    plt.figure(figsize=(14, 4))
    plt.plot(df[DATE_COL], df[noaa_col], label="NOAA", linewidth=1.2)
    plt.plot(df[DATE_COL], df[nasa_col], label="NASA", linewidth=1.2)

    plt.title(f"Daily Precipitation – {station_dir.name}")
    plt.xlabel("Date")
    plt.ylabel("Precipitation (mm/day)")
    plt.legend()
    plt.grid(alpha=0.3)

    out_path = station_dir / FIG_NAME
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"  -> Saved: {out_path}")

print("DONE")

