import xarray as xr

ds = xr.open_dataset(
    r"C:\BKHN\Data Science\subset_GPM_3IMERGDE_2015_2024\3B-DAY.MS.MRG.3IMERG.20150101-S000000-E235959.V07B.nc4"
)

print("Latitude:", ds["lat"].min().item(), "→", ds["lat"].max().item())
print("Longitude:", ds["lon"].min().item(), "→", ds["lon"].max().item())
