#!/usr/bin/env python
# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')
from netCDF4 import Dataset
import numpy as np
import sys
import codecs
import glob
import matplotlib
import matplotlib.pyplot as plt
import math
import cartopy.crs as ccrs
import xarray as xr
import dask as da
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import shap
from sklearn.linear_model import LinearRegression, TheilSenRegressor
from matplotlib import rc
import os
from datetime import datetime
from matplotlib.dates import date2num

import statsmodels.api as sm

lowess = sm.nonparametric.lowess

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
        path="...your_path.../file_to_upload",
        # path="/Net/Groups/BGI/scratch/wantong/study4/programming/Code_Archive_WL/file_to_upload",
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

def heat_map(x_group, y_group, x_axis):
    data = np.zeros((5, 4)) * np.nan
    num = np.zeros((5, 4), dtype=int)
    for y in range(5):
        c = y_group[y,:,:]
        for x in range(4):
            with np.errstate(invalid='ignore'):
                mask = c[np.where((x_group >= x_axis[x]) & (x_group <= x_axis[x + 1]))]
                num[y,x] = np.sum(~np.isnan(mask))

                if np.sum(~np.isnan(mask))>=20:
                    data[y, x] = np.nanmedian(mask)
    return(data,num)

def heatmap(ax, data, ytick, xtick, ticks,colorbar_tick, **kwargs):
    im1 = ax.imshow(data, origin='lower', **kwargs)
    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5)
    ax.set_yticks([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5])
    ax.set_xticklabels(xtick)
    ax.set_yticklabels(ytick)
    cbar = ax.figure.colorbar(im1, ax=ax, ticks=ticks, extend='both', shrink=0.2,aspect=10)
    cbar.ax.set_yticklabels(colorbar_tick)

    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center", rotation_mode="anchor")

    return(im1)


def heat_map1(x_group, data, iqr5, iqr95, x_axis):
    data_median = np.zeros((6, 4)) * np.nan
    num = np.zeros((6, 4), dtype=int)
    num_sig_ratio = np.zeros((6, 4)) * np.nan
    sig = np.zeros((6, 4), dtype=int)

    for y in range(6):
        for x in range(4):
            with np.errstate(invalid='ignore'):
                mask_data = data[y,:,:][np.where((x_group >= x_axis[x]) & (x_group <= x_axis[x + 1]))]
                mask_iqr5 = iqr5[y,:,:][np.where((x_group >= x_axis[x]) & (x_group <= x_axis[x + 1]))]
                mask_iqr95 = iqr95[y,:,:][np.where((x_group >= x_axis[x]) & (x_group <= x_axis[x + 1]))]
                num[y,x] = np.sum(~np.isnan(mask_data))

                if num[y,x]>=20:
                    data_median[y, x] = np.nanmedian(mask_data)

                    num_sig1 = np.sum(~np.isnan(mask_data[mask_data<mask_iqr5]))
                    num_sig2 = np.sum(~np.isnan(mask_data[mask_data>mask_iqr95]))
                    num_sig_ratio[y,x] = (num_sig1+num_sig2)/num[y,x]
                    print(num_sig_ratio[y,x])

                    if num_sig_ratio[y,x] > 0.6:
                        sig[y, x] = 1

    return(data_median, num, sig)

def heatmap1(ax, num, sig, data, ytick, xtick, ticks,colorbar_tick, **kwargs):
    data[0,:] = np.nan
    im1 = ax.imshow(data, origin='lower', **kwargs)
    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5)
    ax.set_yticks([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
    ax.set_xticklabels(xtick)
    ax.set_yticklabels(ytick)
    cbar = ax.figure.colorbar(im1, ax=ax, ticks=ticks, extend='both', shrink=0.2,aspect=10)
    cbar.ax.set_yticklabels(colorbar_tick)

    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center", rotation_mode="anchor")

    for y in range(1,6,1):
        for x in range(4):
            if sig[y, x] == 1:
                ax.scatter(x, y, s=0.1, c='black', marker='s')

    for x in range(4):
        ax.text(x, 0, np.round(num[0, x],2), ha="center", va="center", color="black", fontsize=3)
    return(im1)

def find_drought(drought_ind):
    drought_ind_123 = np.zeros((3)) * np.nan

    year = date[drought_ind].year
    if year == 2018:
        dayofyear = drought_ind + 1
        drought_ind_123[0] = 37 + dayofyear
        drought_ind_123[1] = 83 + dayofyear
        drought_ind_123[2] = 128 + dayofyear

    if year == 2019:
        dayofyear = drought_ind - 37
        drought_ind_123[0] = 0 + dayofyear
        drought_ind_123[1] = 83 + dayofyear
        drought_ind_123[2] = 128 + dayofyear

    if year == 2020:
        dayofyear = drought_ind - 83
        drought_ind_123[0] = 0 + dayofyear
        drought_ind_123[1] = 37 + dayofyear
        drought_ind_123[2] = 128 + dayofyear

    if year == 2021:
        dayofyear = drought_ind - 128
        drought_ind_123[0] = 0 + dayofyear
        drought_ind_123[1] = 37 + dayofyear
        drought_ind_123[2] = 83 + dayofyear

    return(drought_ind_123)

############################### main function

# ###### Fig3 a-c #########
# xticks = [0, 0.8, 1.2, 1.6, 4]
# yticks = [0,1,2,3,4]
# xtick = ['0','0.8','1.2','1.6','>1.6']
# ytick = ['-3 months','-1 months','-8 days','+8 days','+1 months','+3 months']
# aridity = read_data(data_path('aridity_ERA5-land_2018-2020-monthly-mean_0d25.npy'))
# aridity[aridity<0] = np.nan
# aridity[aridity>4] = 4
#
# var = [[-0.08, 0.08], [-4.5, 4.5], [-0.012, 0.012]]
# var_name1 = ['SIF_rel', 'ET_SSEB_final', 'VOD_ratio']
# var_name2 = ['SIFrel', 'ET ($\mathregular{Wm^{-2}}$)', 'VOD ratio']
#
# #####  plotting heatmap ########
# fig = plt.figure(figsize=(3,1.5), dpi=300, tight_layout=True)
# veg = veg_filter(0)
# irrigation = read_data(data_path('gmia_v5_aei_pct_720_1440.npy')) # irrigation mask to remove highly irrigated areas
# drought = read_data(data_path('Drought_ind_SM123_ERA5-Land.npy'))
#
# ## heatmap
# tit = ['(a) ', '(b) ', '(c) ']
# delta = np.zeros((3, 24, 720, 1440)) * np.nan
# delta_ = np.zeros((3, 24, 720, 1440)) * np.nan
# delta1 = np.zeros((3, 5, 720, 1440)) * np.nan
# for tt in [0,1,2]:
#     SIF_8d_ano = read_data(data_path(var_name1[tt] + '_8d_ano_globeDrought.npy'))
#     for row in range(720):
#         for col in range(1440):
#             if ~np.isnan(drought[row, col]) and veg[row, col] >= 0.05 and irrigation[row, col] <= 10:
#                 drought_ind = drought[row, col]
#                 if drought_ind > 12 and drought_ind < 167 - 12:
#                     SIF_pixel = SIF_8d_ano[:, row,col]
#                     delta[tt, :, row, col] = SIF_pixel[int(drought_ind - 12):int(drought_ind + 12)]
#
#
# for tt,pic in zip([0,1,2], [1,2,3]):
#     delta[tt, :, :, :][np.isnan(delta[0, :, :, :])] = np.nan
#     delta[tt, :, :, :][np.isnan(delta[1, :, :, :])] = np.nan
#     delta[tt, :, :, :][np.isnan(delta[2, :, :, :])] = np.nan
#
#     for i, move_window1, move_window2 in zip(range(5), [-12, -4, -1, 1, 4], [-4, -1, 1, 4, 12]):
#         delta1[tt, i, :, :] = np.nanmedian(delta[tt,int(12 + move_window1):int(12 + move_window2),:,:], axis=0)
#
#     for m in range(5):
#         delta1[tt, m, :, :][np.isnan(delta1[tt, 0, :, :])] = np.nan
#         delta1[tt, m, :, :][np.isnan(delta1[tt, 1, :, :])] = np.nan
#         delta1[tt, m, :, :][np.isnan(delta1[tt, 2, :, :])] = np.nan
#         delta1[tt, m, :, :][np.isnan(delta1[tt, 3, :, :])] = np.nan
#         delta1[tt, m, :, :][np.isnan(delta1[tt, 4, :, :])] = np.nan
#
#     # drawing heatmap
#     heatmap_data,num = heat_map(aridity, delta1[tt,:,:,:], xticks)
#     print(np.nanmax(heatmap_data),np.nanmin(heatmap_data))
#     norm = matplotlib.colors.TwoSlopeNorm(vmin=var[tt][0], vcenter=0,vmax=var[tt][1])
#     ticks = [var[tt][0],0,var[tt][1]]
#     colorbar_tick = [var[tt][0],0,var[tt][1]]
#
#     ax = fig.add_subplot(1, 3, pic)
#     heatmap(ax, heatmap_data, ytick, xtick, ticks, colorbar_tick, norm=norm, cmap=plt.get_cmap('coolwarm_r'))
#
#     if pic==1:
#         ax.set_yticklabels(ytick)
#         ax.set_ylabel('Drought period')
#     else:
#         ax.get_yaxis().set_visible(False)
#
#     ax.set_xlabel('Aridity')
#     ax.set_title(tit[pic-1] + var_name2[tt])
#
# plt.savefig(data_path('test/Fig3a-c.jpg'),bbox_inches='tight')



###### Fig3 d-f #########
yticks = [0,1,2,3,4,5]
ytick = ['Physio/Total','-3 months','-1 months','-8 days','+8 days','+1 months','+3 months']
xticks = [0, 0.8, 1.2, 1.6, 4]
xtick = ['0','0.8','1.2','1.6','>1.6']
aridity = read_data(data_path('aridity_ERA5-land_2018-2020-monthly-mean_0d25.npy'))
aridity[aridity<0] = np.nan
aridity[aridity>4] = 4

var = [[-0.08, 0.08], [-4.5, 4.5], [-0.012, 0.012]]
var_name1 = ['SIF_rel', 'ET_SSEB_final', 'VOD_ratio']
var_name2 = ['SIFrel physio', 'ET physio ($\mathregular{Wm^{-2}}$)', 'VOD ratio physio']

#####  plotting heatmap ########
fig = plt.figure(figsize=(3,1.5), dpi=300, tight_layout=True)
veg = veg_filter(0)
irrigation = read_data(data_path('gmia_v5_aei_pct_720_1440.npy')) # irrigation mask to remove highly irrigated areas
drought = read_data(data_path('Drought_ind_SM123_ERA5-Land.npy'))

###### find drought indices for the other 3 years in the same (day-of-year)
SIF_8d = read_data(data_path('test/VDano_physio_SIF_rel_LeaveOut24_LAI.npy'))
date = pd.date_range('2018.03.07', freq='8D', periods=167)
drought123 = np.zeros((3,720,1440))*np.nan
for row in range(720):
    for col in range(1440):
        if veg[row,col]>0.05 and irrigation[row,col]<10:
            if ~np.isnan(drought[row, col]) and ~np.isnan(SIF_8d[:, row, col]).all():

                drought123[:, row, col] = find_drought(drought[row, col])

                # print(drought[row, col], drought123[:, row, col])

tit = ['(d) ', '(e) ', '(f) ']
delta = np.zeros((3, 24, 720, 1440)) * np.nan
delta_ratio = np.zeros((3, 24, 720, 1440)) * np.nan
delta_otheryear = np.zeros((3, 3, 5, 24, 720, 1440)) * np.nan # 3 variables x 3 years x 5 sourrounding 8-days x 24 smooth window x lat x lon
for tt in [0,1,2]:
    SIF_8d_ano = read_data(data_path('test/VDano_physio_' + var_name1[tt] + '_LeaveOut24_LAI.npy'))
    SIF_8d_ano_struc = read_data(data_path('test/VDano_struc_' + var_name1[tt] + '_LeaveOut24_LAI.npy'))
    SIF_8d_ano_ratio = SIF_8d_ano/(SIF_8d_ano_struc+SIF_8d_ano) ## physio/total

    for row in range(720):
        for col in range(1440):
            if ~np.isnan(drought[row, col]) and veg[row, col] >= 0.05 and irrigation[row, col] <= 10:
                if drought[row, col] > 12 and drought[row, col] < 167 - 12:

                    SIF_pixel = SIF_8d_ano[:, row,col]  # 0-137: 2018-03-07 - 2021-02-27; # 0-167: 2018-03-07 - 2021-10-25
                    delta[tt, :, row, col] = SIF_pixel[int(drought[row, col] - 12):int(drought[row, col] + 12)]
                    delta_ratio[tt, :, row, col] = SIF_8d_ano_ratio[int(drought[row, col] - 12):int(drought[row, col] + 12), row, col]

                    ###### load data from the same seasons in non-drought years
                    for m, mov in zip(range(5), [-2,-1,0,1,2]):
                        ind = drought123[0,row, col]
                        if ind > 12+2 and ind < 167 - 12 -2:
                            delta_otheryear[tt, 0, mov, :, row, col] = SIF_pixel[int(ind - 12 + mov):int(ind + 12 + mov)]

                        ind = drought123[1,row, col]
                        if ind > 12+2 and ind < 167 - 1 -2:
                            delta_otheryear[tt, 1, mov, :, row, col] = SIF_pixel[int(ind - 12 + mov):int(ind + 12 + mov)]

                        ind = drought123[2,row, col]
                        if ind > 12+2 and ind < 167 - 12-2:
                            delta_otheryear[tt, 2, mov, :, row, col] = SIF_pixel[int(ind - 12 + mov):int(ind + 12 + mov)]


delta1 = np.zeros((3, 6, 720, 1440)) * np.nan
delta1_otheryear = np.zeros((3, 3, 5, 6, 720, 1440)) * np.nan # 3 variables x 3 years x 5 sourrounding 8-days x 6 smooth window x lat x lon
IQR5 = np.zeros((3, 6, 720, 1440)) * np.nan
IQR95 = np.zeros((3, 6, 720, 1440)) * np.nan
for tt,pic in zip([0,1,2], [1,2,3]):
    delta[tt, :, :, :][np.isnan(delta[0, :, :, :])] = np.nan
    delta[tt, :, :, :][np.isnan(delta[1, :, :, :])] = np.nan
    delta[tt, :, :, :][np.isnan(delta[2, :, :, :])] = np.nan
    delta_ratio[tt, :, :, :][np.isnan(delta[0, :, :, :])] = np.nan


    for i, move_window1, move_window2 in zip(range(6), [-12, -12, -4, -1, 1, 4], [12, -4, -1, 1, 4, 12]):
        delta1_otheryear[tt, :, :, i, :, :] = np.nanmedian(delta_otheryear[tt,:,:,int(12 + move_window1):int(12 + move_window2),:,:], axis=2)
        if i==0:
            delta1[tt, i, :, :] = np.nanmedian(delta_ratio[tt, int(12 + move_window1):int(12 + move_window2), :, :], axis=0)
        else:
            delta1[tt, i, :, :] = np.nanmedian(delta[tt, int(12 + move_window1):int(12 + move_window2), :, :], axis=0)

        ###### randomly sampling (this step takes an hour)
        for row in range(720):
            for col in range(1440):
                sample = delta1_otheryear[tt, :, :, i, row, col].reshape(-1)
                sample1000 = np.random.choice(sample, 1000, replace=True)
                IQR5[tt, i, row, col] = np.nanpercentile(sample1000, 25)
                IQR95[tt, i, row, col] = np.nanpercentile(sample1000, 75)

    np.save(data_path('test/fig3_IQR5_5surrounding_')+ var_name1[tt], IQR5[tt, :,:,:])
    np.save(data_path('test/fig3_IQR5_95surrounding_')+ var_name1[tt], IQR95[tt, :,:,:])

    # IQR5[tt, :,:,:] = read_data(data_path('test/fig3_IQR5_5surrounding_')+ var_name1[tt] +'.npy')
    # IQR95[tt, :,:,:] = read_data(data_path('test/fig3_IQR95_5surrounding_')+ var_name1[tt] +'.npy')

    for m in range(6):
        delta1[tt, m, :, :][np.isnan(delta1[tt, 1, :, :])] = np.nan
        delta1[tt, m, :, :][np.isnan(delta1[tt, 2, :, :])] = np.nan
        delta1[tt, m, :, :][np.isnan(delta1[tt, 3, :, :])] = np.nan
        delta1[tt, m, :, :][np.isnan(delta1[tt, 4, :, :])] = np.nan
        delta1[tt, m, :, :][np.isnan(delta1[tt, 5, :, :])] = np.nan

        IQR5[tt, :, :, :][np.isnan(delta1[tt, :, :, :])] = np.nan
        IQR95[tt, :, :, :][np.isnan(delta1[tt, :, :, :])] = np.nan

    IQR5[tt, 0, :, :] = np.nan
    IQR95[tt, 0, :, :] = np.nan

    # drawing heatmap
    heatmap_data,num,sig = heat_map1(aridity, delta1[tt,:,:,:], IQR5[tt,:,:,:], IQR95[tt,:,:,:], xticks)
    print(np.nanmax(heatmap_data),np.nanmin(heatmap_data))
    norm = matplotlib.colors.TwoSlopeNorm(vmin=var[tt][0], vcenter=0,vmax=var[tt][1])
    ticks = [var[tt][0],0,var[tt][1]]
    colorbar_tick = [var[tt][0],0,var[tt][1]]

    ax = fig.add_subplot(1, 3, pic)
    heatmap1(ax, heatmap_data.copy(), sig, heatmap_data, ytick, xtick, ticks,colorbar_tick, norm=norm, cmap=plt.get_cmap('coolwarm_r'))

    if pic==1:
        ax.set_yticklabels(ytick)
        ax.set_ylabel('Drought period')
    else:
        ax.get_yaxis().set_visible(False)

    ax.set_xlabel('Aridity')
    ax.set_title(tit[pic-1] + var_name2[tt])

plt.savefig(data_path('test/Fig3d-f.jpg'),bbox_inches='tight')

