from pathlib import Path
import pandas as pd


ROOT = Path(r"C:\BKHN\Data Science")

IN_DIR  = ROOT / "NOAA_filled"
OUT_DIR = ROOT / "NOAA_clean"

OUT_DIR.mkdir(exist_ok=True)

PRCP_FILLED_CANDIDATES = [
    "PRCP_FILLED",
    "precip_filled",
    "PRCP_filled",
]

ROUND_DECIMALS = 3

for csv_file in sorted(IN_DIR.glob("*_filled.csv")):
    df = pd.read_csv(csv_file)

    prcp_filled_col = None
    for c in PRCP_FILLED_CANDIDATES:
        if c in df.columns:
            prcp_filled_col = c
            break

    if prcp_filled_col is None:
        print(f"Skip {csv_file.name} (no filled precip column)")
        continue

    if "LATITUDE" in df.columns:
        df["LATITUDE"] = df["LATITUDE"].ffill().bfill()
    if "LONGITUDE" in df.columns:
        df["LONGITUDE"] = df["LONGITUDE"].ffill().bfill()

    df["LATITUDE"]  = df["LATITUDE"].round(ROUND_DECIMALS)
    df["LONGITUDE"] = df["LONGITUDE"].round(ROUND_DECIMALS)
    df[prcp_filled_col] = df[prcp_filled_col].round(ROUND_DECIMALS)

    out = pd.DataFrame({
        "DATE": df["DATE"],
        "LATITUDE": df["LATITUDE"],
        "LONGITUDE": df["LONGITUDE"],
        "PRCP": df[prcp_filled_col],
    })

    out_path = OUT_DIR / csv_file.name.replace("_filled.csv", ".csv")
    out.to_csv(out_path, index=False)

    print(f"Saved {out_path}")

print("DONE")

