import pandas as pd
import geopandas as gpd
import os
import csv
import warnings

warnings.filterwarnings('ignore')

CARAVAN_ROOT_DIR = r"G:\My Drive\DATASETS\CARAVAN\Caravan"
OUTPUT_DIR = r"C:\Users\longh\OneDrive\Attachments\Project\DS Project\dataset_caravan_processed_final"

os.makedirs(OUTPUT_DIR, exist_ok=True)
print("--- STARTING DATA PROCESSING PIPELINE ---")

# ================= STATIC ATTRIBUTES =================

STATIC_SCHEMA = [
    'grdc_no', 'area', 'mean_elevation', 'mean_slope',
    'clay_fraction', 'sand_fraction', 'silt_fraction',
    'forest_cover', 'urban_cover', 'cropland_cover',
    'aridity_index', 'mean_precip'
]

dataset_groups = {}
for root, _, files in os.walk(CARAVAN_ROOT_DIR):
    for file in files:
        if file.startswith("attributes_") and file.endswith(".csv"):
            parts = file.replace('.csv', '').split('_')
            if len(parts) >= 3:
                key = "_".join(parts[2:])
                dataset_groups.setdefault(key, []).append(os.path.join(root, file))

all_dfs = []
for _, file_list in dataset_groups.items():
    merged_df = None
    for f in file_list:
        try:
            df = pd.read_csv(f)
            cid = 'gauge_id' if 'gauge_id' in df.columns else 'grdc_no'
            if cid in df.columns:
                df = df.rename(columns={cid: 'gauge_id'})
                df['gauge_id'] = df['gauge_id'].astype(str)
                merged_df = df if merged_df is None else pd.merge(
                    merged_df, df, on='gauge_id', how='outer'
                )
        except:
            pass
    if merged_df is not None:
        all_dfs.append(merged_df)

if all_dfs:
    df_static = pd.concat(all_dfs, ignore_index=True).drop_duplicates('gauge_id')
    df_static = df_static.rename(columns={
        'gauge_id': 'grdc_no',
        'ele_mt_sav': 'mean_elevation',
        'slp_dg_sav': 'mean_slope',
        'cly_pc_sav': 'clay_fraction',
        'snd_pc_sav': 'sand_fraction',
        'slt_pc_sav': 'silt_fraction',
        'for_pc_sse': 'forest_cover',
        'urb_pc_sse': 'urban_cover',
        'crp_pc_sse': 'cropland_cover',
        'ari_ix_sav': 'aridity_index',
        'pre_mm_syr': 'mean_precip'
    })

    for c in STATIC_SCHEMA:
        if c not in df_static.columns:
            df_static[c] = None

    df_static = df_static[STATIC_SCHEMA]
    df_static.to_csv(os.path.join(OUTPUT_DIR, "step5_static.csv"), index=False)

# ================= UPSTREAM SHAPEFILES =================

UPSTREAM_SCHEMA = ['grdc_no', 'hybas_id', 'is_local', 'sub_area', 'geometry_wkt', 'sub_area_m2']
upstream_path = os.path.join(OUTPUT_DIR, "step5_upstream.csv")

station_to_hybas = {}
hybas_variants = ['hybas_id', 'pfas_id', 'pfaf_id', 'hydroid', 'basin_id']

for root, _, files in os.walk(CARAVAN_ROOT_DIR):
    for file in files:
        if file.startswith("attributes_") and file.endswith(".csv"):
            try:
                path = os.path.join(root, file)
                df_head = pd.read_csv(path, nrows=1)
                cid = 'gauge_id' if 'gauge_id' in df_head.columns else 'grdc_no'
                chybas = next((c for c in df_head.columns if c.lower() in hybas_variants), None)
                if cid and chybas:
                    df_map = pd.read_csv(path, usecols=[cid, chybas])
                    df_map[cid] = df_map[cid].astype(str)
                    station_to_hybas.update(
                        df_map.set_index(cid)[chybas].dropna().to_dict()
                    )
            except:
                pass

if os.path.exists(upstream_path):
    os.remove(upstream_path)

is_header = False
shp_col_map = {
    'grdc_no': ['gauge_id', 'grdc_no', 'station_id', 'id'],
    'sub_area': ['sub_area', 'area', 'shape_area', 'area_sqkm', 'area_m2', 'st_area'],
    'is_local': ['is_local', 'local']
}

for root, _, files in os.walk(CARAVAN_ROOT_DIR):
    for file in files:
        if file.endswith(".shp"):
            try:
                gdf = gpd.read_file(os.path.join(root, file))
                gdf.columns = [c.lower().strip() for c in gdf.columns]

                for target, variants in shp_col_map.items():
                    if target not in gdf.columns:
                        for v in variants:
                            if v in gdf.columns:
                                gdf = gdf.rename(columns={v: target})
                                break

                if 'grdc_no' not in gdf.columns:
                    gdf['grdc_no'] = os.path.splitext(file)[0]
                gdf['grdc_no'] = gdf['grdc_no'].astype(str)

                if 'hybas_id' not in gdf.columns:
                    gdf['hybas_id'] = gdf['grdc_no'].map(station_to_hybas)
                else:
                    gdf['hybas_id'] = gdf['hybas_id'].fillna(
                        gdf['grdc_no'].map(station_to_hybas)
                    )

                if 'geometry' in gdf.columns:
                    gdf['geometry_wkt'] = gdf.geometry.apply(
                        lambda x: x.wkt if x else None
                    )
                    if 'sub_area' not in gdf.columns:
                        gdf['sub_area'] = gdf.geometry.area

                if 'is_local' not in gdf.columns:
                    gdf['is_local'] = 1
                if 'sub_area_m2' not in gdf.columns and 'sub_area' in gdf.columns:
                    gdf['sub_area_m2'] = gdf['sub_area']

                for c in UPSTREAM_SCHEMA:
                    if c not in gdf.columns:
                        gdf[c] = None

                pd.DataFrame(gdf[UPSTREAM_SCHEMA]).dropna(
                    subset=['grdc_no']
                ).to_csv(
                    upstream_path,
                    mode='a',
                    header=not is_header,
                    index=False
                )
                is_header = True
            except:
                pass

# ================= TIME SERIES =================

OUTPUT_CONFIG = {
    'dis': {'filename': 'step5_discharge.csv', 'fields': ['date', 'grdc_no', 'q_obs']},
    'raw': {'filename': 'step5_discharge_raw.csv', 'fields': ['date', 'grdc_no', 'q_obs', 'area_m2']},
    'pre': {'filename': 'step5_precip.csv', 'fields': ['grdc_no', 'date', 'prep_NOAA']},
    'ecm': {'filename': 'step5_ecmwf.csv', 'fields': ['date', 'grdc_no', 't2m', 'sp', 'ssr', 'str', 'tp']},
    'era': {'filename': 'step5_era5.csv', 'fields': ['grdc_no', 'date', 't2m', 'tp', 'sp', 'ssr', 'str']}
}

VAR_MAP = {
    'temperature_2m_mean': 't2m',
    'surface_pressure_mean': 'sp',
    'surface_net_solar_radiation_mean': 'ssr',
    'surface_net_thermal_radiation_mean': 'str',
    'total_precipitation_sum': 'tp'
}

writers, handles = {}, {}
for key, cfg in OUTPUT_CONFIG.items():
    f = open(os.path.join(OUTPUT_DIR, cfg['filename']), 'w', newline='', encoding='utf-8')
    handles[key] = f
    w = csv.DictWriter(f, fieldnames=cfg['fields'])
    w.writeheader()
    writers[key] = w

for root, _, files in os.walk(CARAVAN_ROOT_DIR):
    for file in files:
        if not file.endswith(".csv") or file.startswith(("attributes", "step5")):
            continue
        try:
            df = pd.read_csv(os.path.join(root, file))
            if 'date' not in df.columns:
                continue

            df['grdc_no'] = os.path.splitext(file)[0]

            for src, tgt in VAR_MAP.items():
                if src in df.columns:
                    df = df.rename(columns={src: tgt})

            if 'streamflow' in df.columns:
                df = df.rename(columns={'streamflow': 'q_obs'})
            if 'tp' in df.columns and 'prep_NOAA' not in df.columns:
                df['prep_NOAA'] = df['tp']

            all_cols = set()
            for cfg in OUTPUT_CONFIG.values():
                all_cols.update(cfg['fields'])
            for c in all_cols:
                if c not in df.columns:
                    df[c] = None

            rows = df.to_dict('records')

            if df['q_obs'].notna().any():
                writers['dis'].writerows(
                    [{k: r[k] for k in OUTPUT_CONFIG['dis']['fields']} for r in rows]
                )
                writers['raw'].writerows(
                    [{k: r[k] for k in OUTPUT_CONFIG['raw']['fields']} for r in rows]
                )

            writers['pre'].writerows(
                [{k: r[k] for k in OUTPUT_CONFIG['pre']['fields']} for r in rows]
            )
            writers['ecm'].writerows(
                [{k: r[k] for k in OUTPUT_CONFIG['ecm']['fields']} for r in rows]
            )
            writers['era'].writerows(
                [{k: r[k] for k in OUTPUT_CONFIG['era']['fields']} for r in rows]
            )
        except:
            pass

for f in handles.values():
    f.close()

print("--- PIPELINE FINISHED SUCCESSFULLY ---")
print(f"Output directory: {OUTPUT_DIR}")
