from pathlib import Path
import pandas as pd
import numpy as np
import xarray as xr
import datetime as dt
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import warnings
import math

ROOT = Path(r"C:\BKHN\Data Science")
NOAA_ROOT = ROOT / "NOAA"        
GPM_ROOT = ROOT / "subset_GPM_3IMERGDE_2015_2024"
RESULTS = ROOT / "results_gpm_validation_sa"
RESULTS.mkdir(exist_ok=True)

DATE_START = pd.Timestamp("2015-01-01")
DATE_END   = pd.Timestamp("2024-12-31")
FULL_INDEX = pd.date_range(DATE_START, DATE_END, freq="D")

PPT_VAR_CANDIDATES = ["precipitationCal", "precipitation", "precip"]

def find_gpm_files(gpm_root: Path):
    files = sorted(gpm_root.glob("*.nc4")) + sorted(gpm_root.glob("*.nc"))
    mapping = {}
    import re
    for f in files:
        m = re.search(r"(\d{8})", f.name)
        if m:
            date = dt.datetime.strptime(m.group(1), "%Y%m%d").date()
            mapping[date] = f
    return mapping

GPM_MAP = find_gpm_files(GPM_ROOT)
print(f"Found {len(GPM_MAP)} GPM files")

def detect_ppt_var(ds: xr.Dataset):
    for c in PPT_VAR_CANDIDATES:
        if c in ds.variables:
            return c
    for v in ds.data_vars:
        if ds[v].ndim >= 2:
            return v
    raise KeyError("No precipitation variable found")

def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = map(math.radians, [lat1, lat2])
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1-a))

GPM_DS_CACHE = {}

def get_nasa_nearest(gpm_file, lat, lon):
    key = str(gpm_file)
    if key not in GPM_DS_CACHE:
        ds = xr.open_dataset(gpm_file, mask_and_scale=True)
        GPM_DS_CACHE[key] = {
            "ds": ds,
            "var": detect_ppt_var(ds)
        }

    ds = GPM_DS_CACHE[key]["ds"]
    var = GPM_DS_CACHE[key]["var"]
    da = ds[var]

    latn = next(c for c in da.coords if "lat" in c.lower())
    lonn = next(c for c in da.coords if "lon" in c.lower())

    try:
        sel = da.sel({latn: lat, lonn: lon}, method="nearest")
        val = float(np.squeeze(sel.values))
        glat = float(np.squeeze(sel[latn].values))
        glon = float(np.squeeze(sel[lonn].values))
        dist = haversine_km(lat, lon, glat, glon)
        return val, dist
    except Exception:
        return np.nan, np.nan

def read_noaa_station_csv(csv_path: Path):
    df = pd.read_csv(csv_path, low_memory=False)

    date_col = next(c for c in df.columns if "date" in c.lower())
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.set_index(date_col)

    ppt_col = next(c for c in df.columns if "prcp" in c.lower() or "precip" in c.lower())
    lat = next((df[c].iloc[0] for c in df.columns if "lat" in c.lower()), None)
    lon = next((df[c].iloc[0] for c in df.columns if "lon" in c.lower()), None)

    return {
        "id": csv_path.stem,
        "df": df[[ppt_col]].rename(columns={ppt_col: "noaa_ppt"}),
        "lat": lat,
        "lon": lon
    }

def build_daily_series(noaa_rec):
    df = noaa_rec["df"].copy()
    df.index = pd.to_datetime(df.index)
    df = df.loc[DATE_START:DATE_END]

    daily = df.resample("1D").sum(min_count=1)
    daily = daily.reindex(FULL_INDEX)
    return daily

def add_gpm_series(daily, lat, lon):
    nasa_vals, nasa_dists = [], []
    for d in daily.index.date:
        if d in GPM_MAP:
            v, dist = get_nasa_nearest(GPM_MAP[d], lat, lon)
        else:
            v, dist = np.nan, np.nan
        nasa_vals.append(v)
        nasa_dists.append(dist)

    daily["nasa_ppt"] = nasa_vals
    daily["nasa_dist_km"] = nasa_dists
    return daily

def compute_metrics(daily):
    overlap = daily.dropna(subset=["noaa_ppt", "nasa_ppt"])
    if overlap.empty:
        return {"corr": np.nan, "rmse": np.nan, "mbe": np.nan, "n_pairs": 0}

    x, y = overlap["noaa_ppt"], overlap["nasa_ppt"]
    return {
        "corr": np.corrcoef(x, y)[0, 1],
        "rmse": np.sqrt(mean_squared_error(x, y)),
        "mbe": np.mean(y - x),
        "n_pairs": len(x)
    }

def plot_station(daily, sid, metrics, outdir):
    outdir.mkdir(exist_ok=True)

    plt.figure(figsize=(12,4))
    plt.plot(daily.index, daily["noaa_ppt"], label="NOAA")
    plt.plot(daily.index, daily["nasa_ppt"], label="GPM", alpha=0.8)
    plt.legend()

    plt.title(
        f"{sid} | N={metrics['n_pairs']} | "
        f"r={metrics['corr']:.2f} | RMSE={metrics['rmse']:.2f}"
    )

    plt.tight_layout()
    plt.savefig(outdir / f"{sid}_timeseries_with_gaps.png")
    plt.close()

    overlap = daily.dropna(subset=["noaa_ppt", "nasa_ppt"])
    if not overlap.empty:
        plt.figure(figsize=(5,5))
        plt.scatter(overlap["nasa_ppt"], overlap["noaa_ppt"], s=10)
        m = max(overlap.max().max(), 1)
        plt.plot([0,m],[0,m],'k--')
        plt.xlabel("GPM")
        plt.ylabel("NOAA")
        plt.tight_layout()
        plt.savefig(outdir / f"{sid}_scatter.png")
        plt.close()

def process_all():
    summary = []

    for csv_path in sorted(NOAA_ROOT.glob("*.csv")):
        try:
            rec = read_noaa_station_csv(csv_path)
        except Exception as e:
            warnings.warn(f"Skip {csv_path}: {e}")
            continue

        if rec["lat"] is None or rec["lon"] is None:
            warnings.warn(f"Missing lat/lon for {rec['id']}")
            continue

        daily = build_daily_series(rec)
        daily = add_gpm_series(daily, rec["lat"], rec["lon"])

        metrics = compute_metrics(daily)

        outdir = RESULTS / f"{rec['id']}_validation"
        outdir.mkdir(parents=True, exist_ok=True)

        daily.reset_index().rename(columns={"index":"DATE"}).to_csv(
            outdir / f"{rec['id']}_aligned_daily.csv", index=False
        )

        pd.DataFrame([{
            "station": rec["id"],
            "corr": metrics["corr"],
            "rmse": metrics["rmse"],
            "mbe": metrics["mbe"],
            "n_overlap_days": metrics["n_pairs"]
        }]).to_csv(
            outdir / f"{rec['id']}_metrics.csv",
            index=False
        )

        plot_station(daily, rec["id"], metrics, outdir)

        summary.append({
            "station": rec["id"],
            "corr": metrics["corr"],
            "rmse": metrics["rmse"],
            "mbe": metrics["mbe"],
            "n_pairs": metrics["n_pairs"]
        })

        print(
            f"Done {rec['id']} | "
            f"N={metrics['n_pairs']} | "
            f"r={metrics['corr']:.3f} | "
            f"RMSE={metrics['rmse']:.2f} | "
            f"MBE={metrics['mbe']:.2f}"
        )

    pd.DataFrame(summary).to_csv(
        RESULTS / "validation_summary_global.csv",
        index=False
    )

    for v in GPM_DS_CACHE.values():
        v["ds"].close()

if __name__ == "__main__":
    process_all()

