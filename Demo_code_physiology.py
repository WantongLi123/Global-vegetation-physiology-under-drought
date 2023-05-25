#!/usr/bin/env python
# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import sys
import codecs
import os
import glob
import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt
import math
import cartopy.crs as ccrs
import xarray as xr
import dask as da
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import shap
from sklearn.linear_model import LinearRegression, TheilSenRegressor
from datetime import datetime
from matplotlib.dates import date2num
import statsmodels.api as sm
import scipy.stats as stats
from pathos.multiprocessing import ProcessingPool as Pool
import time

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREAD"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial']})
plt.rcParams.update({'font.size': 4})
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['lines.linewidth'] = 0.8
plt.rcParams['lines.markersize'] = 0.005


def data_path(filename):
    file_path = "{path}/{filename}".format(
        # path="...your_path.../file_to_upload",
        path="/Net/Groups/BGI/scratch/wantong/study4/programming/Code_Archive_WL/file_to_upload",
        filename=filename
    )
    return file_path

def read_data(path):
    data = np.load(path, allow_pickle=True)
    print(path, 'read data')
    return data

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def veg_filter(flag):
    landcover = xr.open_mfdataset(data_path('ESACCI-LC-L4-LCCS-Map-300m-P1Y-aggregated-0.250000Deg-2020-v2.1.1.nc'))
    landcover_all = np.zeros((4, 720, 1440), dtype=np.float16) * np.nan
    multi_predictor = np.zeros((4, 720, 1440), dtype=np.float16) * np.nan
    landcover_all[0, :, :] = landcover.Tree_Broadleaf_Evergreen.values
    landcover_all[1, :, :] = landcover.Tree_Broadleaf_Deciduous.values
    landcover_all[2, :, :] = landcover.Tree_Needleleaf_Evergreen.values
    landcover_all[3, :, :] = landcover.Tree_Needleleaf_Deciduous.values
    multi_predictor[0, :, :] = np.nansum(landcover_all[0:4, :, :], axis=0)

    landcover_all[0, :, :] = landcover.Shrub_Broadleaf_Evergreen.values
    landcover_all[1, :, :] = landcover.Shrub_Broadleaf_Deciduous.values
    landcover_all[2, :, :] = landcover.Shrub_Needleleaf_Evergreen.values
    landcover_all[3, :, :] = landcover.Shrub_Needleleaf_Deciduous.values
    multi_predictor[1, :, :] = np.nansum(landcover_all[0:4, :, :], axis=0)
    multi_predictor[2, :, :] = landcover.Natural_Grass.values
    multi_predictor[3, :, :] = landcover.Managed_Grass.values
    veg_cover = np.nansum(multi_predictor, axis=0)
    tree = multi_predictor[0, :, :]
    tree[tree > 0.8] = 0.8

    if flag == 0:
        return (veg_cover)

    if flag == 1:
        return (tree)

    if flag == 2:
        return (multi_predictor)

def rf(VD, predictors, row):
    print(row)
    size = 24 # 24 time steps are the defined leave-out window (192 days); one can test different windows such as 12 and 6 time steps as the paper did

    physio_struc = np.zeros((2, 167, 1440)) * np.nan

    for col in range(1440):
        if ~np.isnan(drought[row, col]) and veg[row,col]>=0.05 and irrigation[row,col] <= 10:
            drought_ind = drought[row, col]

            if drought_ind > 6 and drought_ind < 167 - 6:
                # LAI + climate
                test_data11 = np.zeros((167, 7)) * np.nan
                test_data11[:, 0] = VD[:, col]  # SIFrel, ET, or VOD ratio
                test_data11[:, 1:7] = predictors[:, 0:6, col]  # LAI+climate

                # remove NaN and all zero arrays to train a model
                test_data1 = test_data11[~np.all(test_data11 == 0, axis=1)]
                test_data = test_data1[~np.any(np.isnan(test_data1), axis=1)]

                if len(test_data[:, 0]) >= 30:
                    ### random forest 2 trained by all LAI + climate
                    rf_full = RandomForestRegressor(n_estimators=100, max_features=0.3, n_jobs=1, bootstrap=True,
                                                oob_score=True, random_state=42)
                    rf_full.fit(test_data[:, 1:7], test_data[:, 0])

                    if rf_full.oob_score_ > 0:
                        target_full = VD[:, col]
                        predictors_full = predictors[:, 0:6, col]  # LAI+climate
                        predictors_full[~np.isfinite(predictors_full)] = np.nan

                        # gap fill by mean to ensure each time step has a prediction
                        predictors_full_gapfill = np.where(np.isnan(predictors_full),
                                                           np.ma.array(predictors_full,
                                                                       mask=np.isnan(predictors_full)).mean(),
                                                           predictors_full)

                        # VI_ideal is the total vegetation anomaly after removing potetnial noise
                        VI_ideal = rf_full.predict(predictors_full_gapfill)
                        VI_ideal[np.isnan(target_full)] = np.nan  # If the target variable is NaN, output needs to reset to NaN
                        VI_ideal[np.isnan(predictors_full[:, 0])] = np.nan  # If LAI variable is NaN, output needs to reset to NaN; hydrometeorological reanalysis rarely have NaN values

                        # To predict vegetation anomalies in each leave-out window, data in the window is removed and the remaining data are used to train a random forest model
                        for move in range(0, 167, size):
                            #  leave-out
                            if move < 167-size:
                                x = np.concatenate((predictors[:move, 0, col], predictors[(move + size):, 0, col]))  # random forest 1: the predictor variable is only LAI
                                y = np.concatenate((VD[:move, col], VD[(move + size):, col]))  # SIFrel, ET, or VOD ratio
                            else: # a final window
                                x = predictors[:move, 0, col]
                                y = VD[:move, col]

                            # remove NaN to train a model
                            x_new = x[~np.isnan(x) & ~np.isnan(y)]
                            y_new = y[~np.isnan(x) & ~np.isnan(y)]

                            if len(x_new) >= 15 and ~np.all(x_new == 0) and ~np.all(y_new == 0):
                                x_new = x_new.reshape(-1, 1)
                                rf_move = RandomForestRegressor(n_estimators=100, max_features=0.3, n_jobs=1, bootstrap=True,
                                                                oob_score=True, random_state=42)
                                rf_move.fit(x_new, y_new)  # RF1(LAI_all)

                                # struc for step[move]
                                if move < 167-size:
                                    target_subset = target_full[move:(move + size)]
                                    LAI_subset = predictors_full[move:(move + size), 0]
                                    LAI_subset_gapfill = predictors_full_gapfill[move:(move + size), 0]

                                    # gap fill by mean to ensure each time step has a prediction
                                    LAI_subset_gapfill_reverse = LAI_subset_gapfill.reshape(-1, 1)
                                    physio_struc[1, move:(move + size), col] = rf_move.predict(LAI_subset_gapfill_reverse) # vegetation structure
                                    physio_struc[1, move:(move + size), col][np.isnan(target_subset)] = np.nan   # If the target variable is NaN, output needs to reset to NaN
                                    physio_struc[1, move:(move + size), col][np.isnan(LAI_subset)] = np.nan # If LAI variable is NaN, output needs to reset to NaN; hydrometeorological reanalysis rarely have NaN values
                                    physio_struc[0, move:(move + size), col] = VI_ideal[move:(move + size)] - physio_struc[1, move:(move + size), col] # vegetation physiology

                                else:
                                    target_subset = target_full[move:]
                                    LAI_subset = predictors_full[move:, 0]
                                    LAI_subset_gapfill = predictors_full_gapfill[move:, 0]
                                    LAI_subset_gapfill_reverse = LAI_subset_gapfill.reshape(-1, 1)
                                    physio_struc[1, move:, col] = rf_move.predict(LAI_subset_gapfill_reverse)
                                    physio_struc[1, move:, col][np.isnan(target_subset)] = np.nan
                                    physio_struc[1, move:, col][np.isnan(LAI_subset)] = np.nan
                                    physio_struc[0, move:, col] = VI_ideal[move] - physio_struc[1, move:,col]

    return(physio_struc)

############################## main function
t2m_8d = read_data(data_path('t2m_2018-03-07_2021-10-25_0d25_8d.npy'))[:,::-1,:]
SIF_MSC = read_data(data_path('SIF_rel_8d_seasonality.npy'))
drought = read_data(data_path('Drought_ind_SM123_ERA5-Land.npy'))

veg = veg_filter(0) # vegetation mask to identify the global study area
irrigation = read_data(data_path('gmia_v5_aei_pct_720_1440.npy')) # irrigation mask to remove highly irrigated areas

var_name1 = ['SIF_rel', 'ET_SSEB_final', 'VOD_ratio']
var_name2 = ['LAI', 'tp', 't2m', 'ssrd', 'vpd','swvl123']

predictors_8d_ano = np.zeros((167,6,720,1440)) * np.nan # time x variables x lat x lon; time: 2018-03-07_2021-10-25, 8 daily

# ~ 2 hours min runtime for each variable
start_time = time.time()

vd_ind = 0 # vd_ind = 0 for SIFrel, vd_ind = 1 for ET, vd_ind = 2 for VOD ratio
VD_8d_ano = read_data(data_path(var_name1[vd_ind] + '_8d_ano_globeDrought.npy')) # SIFrel, ET, or VOD ratio
VD_8d_ano[t2m_8d <= 5 + 273.15] = np.nan  # T<5˚C winter season
VD_8d_ano[SIF_MSC <= 0.2] = np.nan  # sif<0.2 non-growing season

for pic in range(6):
    predictors_8d_ano[:, pic, :, :] = read_data(data_path(var_name2[pic] + '_8d_ano_globeDrought.npy'))

# set CPU usage
pool = Pool(16)
VD = [VD_8d_ano[:, part,:] for part in range(720)]
predictors = [predictors_8d_ano[:, :, part,:] for part in range(720)]
row = [part for part in range(720)]

outs = pool.map(rf, VD, predictors, row) # multiprocessing
pool.close()
pool.join()
print('shape(outs):', np.shape(outs))

physio_struc = np.zeros((2, 167, 720, 1440)) * np.nan

for part in range(720):
    physio_struc[:, :, part, :] = outs[part]

np.save(data_path('test/VDano_physio_' + var_name1[vd_ind] + '_LeaveOut24_LAI'),physio_struc[0,:,:,:])  # Random forest
np.save(data_path('test/VDano_struc_' + var_name1[vd_ind] + '_LeaveOut24_LAI'),physio_struc[1,:,:,:])  # Random forest

print("16 cpus--- %s seconds ---" % (time.time() - start_time))


# Visulization
def europe_map(data, ax, min, max):
    ax.coastlines(linewidth=0.1)
    ax.set_extent([-10, 40, 35, 70], crs=ccrs.PlateCarree())
    lon = np.arange(-10, 40, 0.25)
    lat = np.arange(35, 70, 0.25)
    if max > 0 and min < 0:
        norm = matplotlib.colors.TwoSlopeNorm(vmin=min, vcenter=0, vmax=max)
        cmap = 'coolwarm'
    elif max <= 0:
        norm = matplotlib.colors.Normalize(vmin=min, vmax=max)
        cmap = truncate_colormap(plt.get_cmap('coolwarm'), 0, 0.5)
    elif min >= 0:
        norm = matplotlib.colors.Normalize(vmin=min, vmax=max)
        cmap = truncate_colormap(plt.get_cmap('coolwarm'), 0.5, 1)
    map = ax.pcolormesh(lon, lat, data[::-1, :], cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
    cbar = ax.figure.colorbar(map, ax=ax, ticks=[min, max], shrink=0.3, extend='both')

    return (cbar)


anomaly = read_data(data_path('test/VDano_physio_SIF_rel_LeaveOut24_LAI.npy'))
print(np.shape(anomaly))
data1 = np.nanmax(anomaly, axis=0)
data2 = np.nanmin(anomaly, axis=0)
fig = plt.figure(figsize=(5, 4), dpi=300, tight_layout=True)
ax = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
europe_map(data1[80:220, 680:880], ax, min=np.nanpercentile(data1,25), max=np.nanpercentile(data1,75))
ax.set_title('Maximum SIFrel physiology anomalies')
ax = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())
europe_map(data2[80:220, 680:880], ax, min=np.nanpercentile(data2,25), max=np.nanpercentile(data2,75))
ax.set_title('Minimum SIFrel physiology anomalies')
plt.savefig(data_path('test/SIFrel_physiology.jpg'),bbox_inches='tight')
