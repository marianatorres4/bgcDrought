import xarray as xr
import dask
from dask.diagnostics import ProgressBar

nwa_data = "/dfs9/hfdrake_hpc/datasets/NWA/NWA12_COBALT_2025_04_bugfixes/gfdl.ncrc6-intel23-prod-hdf5/pp/"

temp_ds = xr.open_mfdataset(
    f"{nwa_data}ocean_monthly_z/ts/monthly/5yr/*.thetao.nc",
    chunks={}
)

no3_ds = xr.open_mfdataset(
    f"{nwa_data}ocean_cobalt_tracers_month_z/ts/monthly/5yr/*.no3.nc",
    chunks={}
)

tos = temp_ds["thetao"].isel(z_l=0, drop=True)
no3s = no3_ds["no3"].isel(z_l=0, drop=True)

tos_clim = tos.groupby("time.month").mean("time")
no3s_clim = no3s.groupby("time.month").mean("time")

tos_per90 = tos.chunk(dict(time=-1)).groupby("time.month").quantile(0.90, dim="time")
no3s_per10 = no3s.chunk(dict(time=-1)).groupby("time.month").quantile(0.10, dim="time")

tos_clim_ts = tos_clim.sel(month=tos.time.dt.month)
no3s_clim_ts = no3s_clim.sel(month=no3s.time.dt.month)

tos_per90_ts = tos_per90.sel(month=tos.time.dt.month)
no3s_per10_ts = no3s_per10.sel(month=no3s.time.dt.month)
# Rechunk remapped time series
tos_clim_ts = tos_clim_ts.chunk({"time": 12})
no3s_clim_ts = no3s_clim_ts.chunk({"time": 12})

tos_per90_ts = tos_per90_ts.chunk({"time": 12})
no3s_per10_ts = no3s_per10_ts.chunk({"time": 12})

output = "/dfs6b/pub/mariant3/NWA/data/"

with ProgressBar():

    tos_clim_ts.to_netcdf(
        f"{output}tos_monthly_clim_1993-2022.nc"
    )

    no3s_clim_ts.to_netcdf(
        f"{output}no3s_monthly_clim_1993-2022.nc"
    )

    tos_per90_ts.to_netcdf(
        f"{output}tos_monthly_per90_1993-2022.nc"
    )

    no3s_per10_ts.to_netcdf(
        f"{output}no3s_monthly_per10_1993-2022.nc"
    )

