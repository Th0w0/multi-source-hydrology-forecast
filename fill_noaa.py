from pathlib import Path
import pandas as pd
import numpy as np
import xarray as xr
from sklearn.linear_model import LinearRegression

# ==================================================
# CONFIG
# ==================================================
ROOT = Path(r"C:\BKHN\Data Science")

NOAA_DIR = ROOT / "NOAA"
NOAA_FILLED_DIR = ROOT / "NOAA_filled"
GPM_DIR = ROOT / "subset_GPM_3IMERGDE_2015_2024"
STRATEGY_CSV = ROOT / "stations_fill_strategy.csv"

DATE_START = "2015-01-01"
DATE_END   = "2024-12-31"
FULL_INDEX = pd.date_range(DATE_START, DATE_END, freq="D")

NOAA_FILLED_DIR.mkdir(exist_ok=True)

PPT_CANDIDATES = ["precipitationCal", "precipitation", "precip"]

# threshold mưa rất nhỏ (mm/day)
DRY_EPS = 0.2

# ==================================================
# HELPERS
# ==================================================
def is_wet_month(m):
    return m in [10, 11, 12, 1, 2, 3, 4]

def read_gpm_daily(date, lat, lon):
    """
    Read GPM IMERG daily: 1 file = 1 day
    Return scalar precipitation or NaN
    """
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

            return (
                da.sel(
                    {lat_name: lat, lon_name: lon},
                    method="nearest"
                )
                .values
                .item()
            )
    except Exception:
        return np.nan

# ==================================================
# LOAD STRATEGY
# ==================================================
strategy_df = pd.read_csv(STRATEGY_CSV).set_index("station")

# ==================================================
# MAIN LOOP
# ==================================================
for noaa_file in sorted(NOAA_DIR.glob("*.csv")):
    station = noaa_file.stem
    print(f"\nProcessing {station}")

    # ---- skip if not in strategy or DO_NOT_FILL
    if station not in strategy_df.index:
        print("  -> Skip (no strategy)")
        continue

    strategy = strategy_df.loc[station, "strategy"]
    if strategy == "DO_NOT_FILL":
        print("  -> Skip (DO_NOT_FILL)")
        continue

    # ==================================================
    # LOAD NOAA
    # ==================================================
    df = pd.read_csv(noaa_file)

    date_col = next(c for c in df.columns if "date" in c.lower())
    ppt_col  = next(c for c in df.columns if "prcp" in c.lower() or "precip" in c.lower())
    lat_col  = next(c for c in df.columns if "lat" in c.lower())
    lon_col  = next(c for c in df.columns if "lon" in c.lower())

    lat = float(df[lat_col].iloc[0])
    lon = float(df[lon_col].iloc[0])

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    df = df.loc[DATE_START:DATE_END]
    df = df.rename(columns={ppt_col: "noaa"})
    df = df.reindex(FULL_INDEX)
    df.index.name = "DATE"

    # ==================================================
    # READ GPM
    # ==================================================
    df["gpm"] = [
        read_gpm_daily(d, lat, lon)
        for d in df.index
    ]

    # ==================================================
    # TRAIN REGRESSION (IF NEEDED)
    # ==================================================
    a = b = None
    if strategy == "LINEAR_REGRESSION":
        wet_train = df[
            df.index.month.map(is_wet_month)
            & df["noaa"].notna()
            & df["gpm"].notna()
        ]

        if len(wet_train) >= 30:
            model = LinearRegression()
            model.fit(wet_train[["gpm"]], wet_train["noaa"])
            a = model.coef_[0]
            b = model.intercept_
        else:
            strategy = "FILL_PLUS_BIAS"

    # ==================================================
    # COMPUTE WET BIAS (SAFE)
    # ==================================================
    wet_bias = (
        df.loc[
            df.index.month.map(is_wet_month)
            & df["noaa"].notna()
            & df["gpm"].notna(),
            "gpm"
        ]
        -
        df.loc[
            df.index.month.map(is_wet_month)
            & df["noaa"].notna()
            & df["gpm"].notna(),
            "noaa"
        ]
    ).mean()

    if np.isnan(wet_bias):
        wet_bias = 0.0

    # ==================================================
    # FILL LOGIC (WITH THRESHOLD – NO NaN)
    # ==================================================
    filled = []
    method = []

    for d, r in df.iterrows():
        # ---- NOAA exists
        if not pd.isna(r["noaa"]):
            filled.append(r["noaa"])
            method.append("NOAA")
            continue

        # ---- GPM missing
        if pd.isna(r["gpm"]):
            filled.append(0.0)
            method.append("GPM_MISSING_ZERO")
            continue

        gpm_val = r["gpm"]
        wet = is_wet_month(d.month)

        # ---- threshold: very small GPM -> no rain
        if gpm_val < DRY_EPS:
            filled.append(0.0)
            method.append("GPM_BELOW_THRESHOLD")
            continue

        # ---- apply fill
        if wet:
            if strategy == "LINEAR_REGRESSION" and a is not None:
                filled.append(max(0.0, a * gpm_val + b))
                method.append("GPM_LINEAR")
            else:
                filled.append(max(0.0, gpm_val - wet_bias))
                method.append("GPM_BIAS")
        else:
            filled.append(gpm_val)
            method.append("GPM_RAW")

    df["PRCP_FILLED"] = filled
    df["FILL_METHOD"] = method

    out = df.reset_index()
    out_path = NOAA_FILLED_DIR / f"{station}_filled.csv"
    out.to_csv(out_path, index=False)
    print(f"  -> Saved {out_path.name}")

print("\nDONE")
