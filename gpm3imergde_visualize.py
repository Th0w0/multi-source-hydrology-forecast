from pathlib import Path
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature


# =========================
# CONFIG
# =========================
DATA_DIR = Path("Grid_SA")
FILE_NAME = "precip.2024_SA.nc4"
DAY = "2024-01-01"

# BBOX NAM MỸ (CPC: lon 0–360)
LAT_MIN, LAT_MAX = -45.1, 10.94202
LON_MIN, LON_MAX = 281.7375, 316.416667


# =========================
# LOAD DATA
# =========================
ds = xr.open_dataset(DATA_DIR / FILE_NAME)

# auto detect precipitation variable
for v in ["precip", "precipitation"]:
    if v in ds.data_vars:
        precip_var = v
        break
else:
    raise ValueError("Không tìm thấy biến mưa")

precip = ds[precip_var].sel(time=DAY)


# =========================
# BASIC SANITY CHECK
# =========================
print("Min:", float(precip.min()), "Max:", float(precip.max()))


# =========================
# PLOT – SOUTH AMERICA ONLY
# =========================
fig = plt.figure(figsize=(7, 8))

ax = plt.axes(projection=ccrs.PlateCarree())

# ❌ BỎ GLOBAL
# ax.set_global()

# ✅ SET EXTENT NAM MỸ
ax.set_extent(
    [LON_MIN, LON_MAX, LAT_MIN, LAT_MAX],
    crs=ccrs.PlateCarree()
)

# map features
ax.add_feature(cfeature.LAND, facecolor="lightgray")
ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
ax.add_feature(cfeature.BORDERS, linewidth=0.4)

# robust color scale
vmax = np.nanpercentile(precip.values, 95)

im = ax.pcolormesh(
    ds["lon"],
    ds["lat"],
    precip,
    cmap="Blues",
    vmin=0,
    vmax=vmax,
    transform=ccrs.PlateCarree()
)

cb = plt.colorbar(im, ax=ax, shrink=0.75)
cb.set_label("Precipitation (mm/day)")

ax.set_title(
    f"CPC Daily Precipitation – South America\n{DAY}",
    fontsize=11
)

plt.tight_layout()
plt.show()
