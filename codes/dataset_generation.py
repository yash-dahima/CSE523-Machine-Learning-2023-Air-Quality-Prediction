# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 21:56:09 2023

@author: Yash Dahima
"""

import xarray as xr, pandas as pd, os, numpy as np

ds1 = xr.open_mfdataset('C:/Users/Yash Dahima/PhD/Course Work/ML/Project/AQI/Datasets/CAMS Dataset/1/*.nc').mean(dim=['latitude', 'longitude']).sel(time=slice('2011-01-01T00:00:00', '2019-12-31T21:00:00'))
ds2 = xr.open_mfdataset('C:/Users/Yash Dahima/PhD/Course Work/ML/Project/AQI/Datasets/CAMS Dataset/2/*.nc').mean(dim=['latitude', 'longitude'])

ds = xr.merge([ds1, ds2], compat='override')

wind_speed = np.sqrt(np.square(ds.u10)+np.square(ds.v10)) # m/s
wind_direction = (np.rad2deg(np.arctan2(ds.v10, ds.u10))) % 360 # in degrees , Northward = 0, Eastward = 90
dew_temp = ds.d2m - 273.15 # celcius
temp = ds.t2m - 273.15 # celcius
pm2p5 = ds.pm2p5*1e9 # ug/m3
pressure = ds.sp/100 #hPa
water_vapour = ds.tcwv # Total column vertically-integrated water vapour (kg m**-2)


blh = xr.open_mfdataset('C:/Users/Yash Dahima/PhD/Course Work/ML/Project/AQI/Datasets/ERA5 Dataset/*.nc')
blh = blh.sel(expver=1).combine_first(blh.sel(expver=5))
blh = blh.mean(dim=['latitude', 'longitude']).blh
blh = blh.sel(time=slice('2011-01-01T00:00:00', '2019-12-31T23:00:00'))
blh = blh.resample(time='3H').mean()



ml_dataset = pd.DataFrame()

ml_dataset.index, ml_dataset['ws'], ml_dataset['wd'], ml_dataset['temp'], ml_dataset['dew_temp'], ml_dataset['pressure'], ml_dataset['z'], ml_dataset['wv'], ml_dataset['blh'], ml_dataset['bcaod550'], ml_dataset['duaod550'], ml_dataset['omaod550'], ml_dataset['ssaod550'], ml_dataset['suaod550'], ml_dataset['aod469'], ml_dataset['aod550'], ml_dataset['aod670'], ml_dataset['aod865'], ml_dataset['aod1240'], ml_dataset['pm2p5'] = temp.time + pd.Timedelta(hours=5, minutes=30), wind_speed.data, wind_direction.data, temp.data, dew_temp.data, pressure.data, ds.z.data, water_vapour.data, blh.data, ds.bcaod550.data, ds.duaod550.data, ds.omaod550.data, ds.ssaod550.data, ds.suaod550.data, ds.aod469.data, ds.aod550.data, ds.aod670.data, ds.aod865.data, ds.aod1240.data, pm2p5.data # time in IST

ml_dataset.to_excel("C:/Users/Yash Dahima/PhD/Course Work/ML/Project/AQI/Datasets/data.xlsx")




"""
dss
<xarray.Dataset>
Dimensions:    (longitude: 2, latitude: 2, time: 33592)
Coordinates:
  * longitude  (longitude) float32 72.0 72.75
  * latitude   (latitude) float32 23.25 22.5
  * time       (time) datetime64[ns] 2011-01-01 ... 2022-06-30T21:00:00
Data variables:
    u10        (time, latitude, longitude) float32 dask.array<chunksize=(11688, 2, 2), meta=np.ndarray>
    v10        (time, latitude, longitude) float32 dask.array<chunksize=(11688, 2, 2), meta=np.ndarray>
    d2m        (time, latitude, longitude) float32 dask.array<chunksize=(11688, 2, 2), meta=np.ndarray>
    t2m        (time, latitude, longitude) float32 dask.array<chunksize=(11688, 2, 2), meta=np.ndarray>
    pm2p5      (time, latitude, longitude) float32 dask.array<chunksize=(11688, 2, 2), meta=np.ndarray>
    sp         (time, latitude, longitude) float32 dask.array<chunksize=(11688, 2, 2), meta=np.ndarray>
    aod550     (time, latitude, longitude) float32 dask.array<chunksize=(11688, 2, 2), meta=np.ndarray>
    tcwv       (time, latitude, longitude) float32 dask.array<chunksize=(11688, 2, 2), meta=np.ndarray>
Attributes:
    Conventions:  CF-1.6
    history:      2023-02-24 15:25:51 GMT by grib_to_netcdf-2.25.1: /opt/ecmw...
    

blh
<xarray.Dataset>
Dimensions:    (time: 105192, latitude: 5, longitude: 5)
Coordinates:
  * longitude  (longitude) float32 72.0 72.25 72.5 72.75 73.0
  * latitude   (latitude) float32 23.5 23.25 23.0 22.75 22.5
  * time       (time) datetime64[ns] 2011-01-01 ... 2022-12-31T23:00:00
Data variables:
    blh        (time, latitude, longitude) float32 dask.array<chunksize=(52608, 5, 5), meta=np.ndarray>
Attributes:
    Conventions:  CF-1.6
    history:      2023-02-24 17:31:42 GMT by grib_to_netcdf-2.25.1: /opt/ecmw...
"""