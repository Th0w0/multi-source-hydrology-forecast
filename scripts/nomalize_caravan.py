import pandas as pd
import geopandas as gpd
import os
import csv
import warnings

# Tắt cảnh báo CRS để màn hình sạch sẽ
warnings.filterwarnings('ignore')

# ================= CẤU HÌNH HỆ THỐNG =================
CARAVAN_ROOT_DIR = r"G:\My Drive\DATASETS\CARAVAN\Caravan"      # Thư mục nguồn
OUTPUT_DIR = r"C:\Users\longh\OneDrive\Attachments\Project\DS Project\dataset_caravan_processed_final" # Thư mục đích

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"--- KHỞI CHẠY QUY TRÌNH XỬ LÝ DỮ LIỆU TOÀN DIỆN ---")

# ================= PHẦN 1: ATTRIBUTES (DỮ LIỆU TĨNH) =================
print("\n[PHẦN 1/3] XỬ LÝ ATTRIBUTES...")

STATIC_SCHEMA = [
    'grdc_no', 'area', 'mean_elevation', 'mean_slope', 
    'clay_fraction', 'sand_fraction', 'silt_fraction', 
    'forest_cover', 'urban_cover', 'cropland_cover', 
    'aridity_index', 'mean_precip'
]

# 1.1 Quét file attributes
dataset_groups = {}
print("   -> Đang quét file attributes...")
for root, dirs, files in os.walk(CARAVAN_ROOT_DIR):
    for file in files:
        if file.startswith("attributes_") and file.endswith(".csv"):
            parts = file.replace('.csv', '').split('_')
            if len(parts) >= 3:
                suffix = "_".join(parts[2:])
                if suffix not in dataset_groups: dataset_groups[suffix] = []
                dataset_groups[suffix].append(os.path.join(root, file))

# 1.2 Merge dữ liệu
all_dfs = []
for suffix, file_list in dataset_groups.items():
    merged_df = None
    for f in file_list:
        try:
            df = pd.read_csv(f)
            # Chuẩn hóa ID
            col_id = 'gauge_id' if 'gauge_id' in df.columns else 'grdc_no'
            if col_id in df.columns:
                df = df.rename(columns={col_id: 'gauge_id'})
                df['gauge_id'] = df['gauge_id'].astype(str)
                
                if merged_df is None: merged_df = df
                else: merged_df = pd.merge(merged_df, df, on='gauge_id', how='outer')
        except: pass
    if merged_df is not None: all_dfs.append(merged_df)

if all_dfs:
    df_static = pd.concat(all_dfs, ignore_index=True)
    df_static = df_static.drop_duplicates(subset=['gauge_id'])
    
    # Mapping tên cột
    rename_map = {
        'gauge_id': 'grdc_no',
        'area': 'area',
        'ele_mt_sav': 'mean_elevation', 'slp_dg_sav': 'mean_slope',
        'cly_pc_sav': 'clay_fraction', 'snd_pc_sav': 'sand_fraction', 'slt_pc_sav': 'silt_fraction',
        'for_pc_sse': 'forest_cover', 'urb_pc_sse': 'urban_cover', 'crp_pc_sse': 'cropland_cover',
        'ari_ix_sav': 'aridity_index', 'pre_mm_syr': 'mean_precip'
    }
    df_static = df_static.rename(columns=rename_map)
    
    # Đảm bảo đủ cột
    for c in STATIC_SCHEMA:
        if c not in df_static.columns: df_static[c] = None
    
    df_static = df_static[STATIC_SCHEMA]
    df_static.to_csv(os.path.join(OUTPUT_DIR, "step5_static.csv"), index=False)
    print("   -> HOÀN TẤT: step5_static.csv")

# ================= PHẦN 2: SHAPEFILES (FIX UPSTREAM + HYSETS) =================
print("\n[PHẦN 2/3] XỬ LÝ SHAPEFILES (UPSTREAM)...")
UPSTREAM_SCHEMA = ['grdc_no', 'hybas_id', 'is_local', 'sub_area', 'geometry_wkt', 'sub_area_m2']
upstream_path = os.path.join(OUTPUT_DIR, "step5_upstream.csv")

# 2.1 TẠO TỪ ĐIỂN TRA CỨU HYBAS_ID (QUAN TRỌNG ĐỂ FIX LỖI)
print("   -> Đang tạo bảng tham chiếu Hybas_ID từ attributes...")
station_to_hybas = {}
hybas_variants = ['hybas_id', 'pfas_id', 'pfaf_id', 'hydroid', 'basin_id']

for root, dirs, files in os.walk(CARAVAN_ROOT_DIR):
    for file in files:
        if file.startswith("attributes_") and file.endswith(".csv"):
            try:
                # Đọc header
                path = os.path.join(root, file)
                df_head = pd.read_csv(path, nrows=1)
                
                # Tìm tên cột ID và Hybas
                cid = 'gauge_id' if 'gauge_id' in df_head.columns else ('grdc_no' if 'grdc_no' in df_head.columns else None)
                chybas = next((c for c in df_head.columns if c.lower() in hybas_variants), None)
                
                if cid and chybas:
                    df_map = pd.read_csv(path, usecols=[cid, chybas])
                    df_map[cid] = df_map[cid].astype(str)
                    station_to_hybas.update(df_map.set_index(cid)[chybas].dropna().to_dict())
            except: pass
print(f"      ...Đã tìm thấy {len(station_to_hybas)} mã trạm có Hybas_ID.")

# 2.2 XỬ LÝ FILE SHAPEFILE
if os.path.exists(upstream_path): os.remove(upstream_path)
is_header = False

# Mapping tên cột Shapefile
shp_col_map = {
    'grdc_no':   ['gauge_id', 'grdc_no', 'station_id', 'id'],
    'sub_area':  ['sub_area', 'area', 'shape_area', 'area_sqkm', 'area_m2', 'st_area'],
    'is_local':  ['is_local', 'local']
}

for root, dirs, files in os.walk(CARAVAN_ROOT_DIR):
    for file in files:
        if file.endswith(".shp"):
            try:
                gdf = gpd.read_file(os.path.join(root, file))
                gdf.columns = [c.lower().strip() for c in gdf.columns]
                
                # Đổi tên cột thông minh
                for target, vars in shp_col_map.items():
                    if target not in gdf.columns:
                        for v in vars: 
                            if v in gdf.columns: 
                                gdf = gdf.rename(columns={v: target}); break
                
                # Fallback: Nếu không có ID, lấy tên file
                if 'grdc_no' not in gdf.columns:
                    gdf['grdc_no'] = os.path.splitext(file)[0]
                gdf['grdc_no'] = gdf['grdc_no'].astype(str)

                # --- FIX CHÍNH: ĐIỀN HYBAS_ID TỪ TỪ ĐIỂN ---
                # Nếu cột hybas_id thiếu hoặc trống, điền từ station_to_hybas
                if 'hybas_id' not in gdf.columns:
                    gdf['hybas_id'] = gdf['grdc_no'].map(station_to_hybas)
                else:
                    gdf['hybas_id'] = gdf['hybas_id'].fillna(gdf['grdc_no'].map(station_to_hybas))
                
                # Fallback: Tính diện tích
                if 'geometry' in gdf.columns:
                    gdf['geometry_wkt'] = gdf.geometry.apply(lambda x: x.wkt if x else None)
                    if 'sub_area' not in gdf.columns:
                        gdf['sub_area'] = gdf.geometry.area 
                
                if 'is_local' not in gdf.columns: gdf['is_local'] = 1
                if 'sub_area_m2' not in gdf.columns and 'sub_area' in gdf.columns:
                     gdf['sub_area_m2'] = gdf['sub_area']
                
                # Ghi file
                for c in UPSTREAM_SCHEMA:
                    if c not in gdf.columns: gdf[c] = None
                
                df_up = pd.DataFrame(gdf[UPSTREAM_SCHEMA]).dropna(subset=['grdc_no'])
                df_up.to_csv(upstream_path, mode='a', header=not is_header, index=False)
                is_header = True
            except: pass
print("   -> HOÀN TẤT: step5_upstream.csv")

# ================= PHẦN 3: TIME-SERIES (DỮ LIỆU KHÍ TƯỢNG) =================
print("\n[PHẦN 3/3] XỬ LÝ TIME-SERIES (CÓ THỂ MẤT NHIỀU THỜI GIAN)...")

OUTPUT_CONFIG = {
    'dis': {'filename': 'step5_discharge.csv',     'fields': ['date', 'grdc_no', 'q_obs']},
    'raw': {'filename': 'step5_discharge_raw.csv', 'fields': ['date', 'grdc_no', 'q_obs', 'area_m2']},
    'pre': {'filename': 'step5_precip.csv',        'fields': ['grdc_no', 'date', 'prep_NOAA']},
    'ecm': {'filename': 'step5_ecmwf.csv',         'fields': ['date', 'grdc_no', 't2m', 'sp', 'ssr', 'str', 'tp']},
    'era': {'filename': 'step5_era5.csv',          'fields': ['grdc_no', 'date', 't2m', 'tp', 'sp', 'ssr', 'str']}
}

VAR_MAP = {
    'temperature_2m_mean': 't2m', 'surface_pressure_mean': 'sp',
    'surface_net_solar_radiation_mean': 'ssr', 'surface_net_thermal_radiation_mean': 'str',
    'total_precipitation_sum': 'tp'
}

writers, handles = {}, {}
for key, cfg in OUTPUT_CONFIG.items():
    path = os.path.join(OUTPUT_DIR, cfg['filename'])
    h = open(path, 'w', newline='', encoding='utf-8')
    handles[key] = h
    w = csv.DictWriter(h, fieldnames=cfg['fields'])
    w.writeheader()
    writers[key] = w

count = 0
for root, dirs, files in os.walk(CARAVAN_ROOT_DIR):
    for file in files:
        if not file.endswith(".csv") or file.startswith("attributes") or file.startswith("step5"): continue
        try:
            df = pd.read_csv(os.path.join(root, file))
            if 'date' not in df.columns: continue
            
            df['grdc_no'] = os.path.splitext(file)[0]
            
            # Đổi tên biến
            for src, tgt in VAR_MAP.items():
                if src in df.columns: df = df.rename(columns={src: tgt})
            
            if 'streamflow' in df.columns: df = df.rename(columns={'streamflow': 'q_obs'})
            if 'total_precipitation_sum' in df.columns: df = df.rename(columns={'total_precipitation_sum': 'prep_NOAA'})
            # Fix prep_NOAA
            if 'tp' in df.columns and ('prep_NOAA' not in df.columns or df['prep_NOAA'].isnull().all()):
                 df['prep_NOAA'] = df['tp']

            # Điền None cho cột thiếu
            all_cols = set()
            for cfg in OUTPUT_CONFIG.values(): all_cols.update(cfg['fields'])
            for c in all_cols: 
                if c not in df.columns: df[c] = None

            # Ghi dữ liệu
            rows = df.to_dict('records')
            
            # Discharge
            if df['q_obs'].notna().any():
                writers['dis'].writerows([{k: r[k] for k in OUTPUT_CONFIG['dis']['fields']} for r in rows])
                writers['raw'].writerows([{k: r[k] for k in OUTPUT_CONFIG['raw']['fields']} for r in rows])
            
            # Precip & Meteo
            writers['pre'].writerows([{k: r[k] for k in OUTPUT_CONFIG['pre']['fields']} for r in rows])
            writers['ecm'].writerows([{k: r[k] for k in OUTPUT_CONFIG['ecm']['fields']} for r in rows])
            writers['era'].writerows([{k: r[k] for k in OUTPUT_CONFIG['era']['fields']} for r in rows])

            count += 1
            if count % 500 == 0: print(f"   ...Đã xử lý {count} trạm")
        except: pass

for h in handles.values(): h.close()
print(f"\n--- XONG TOÀN BỘ QUY TRÌNH! ---")
print(f"Thư mục kết quả: {OUTPUT_DIR}")