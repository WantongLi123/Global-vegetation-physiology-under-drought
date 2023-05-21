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
from matplotlib.dates import date2num
import math
from datetime import date
from sklearn.metrics import r2_score
import statsmodels.api as sm

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREAD"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Arial']})
plt.rcParams.update({'font.size': 5})
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['lines.linewidth'] = 0.7
plt.rcParams['lines.markersize'] = 0.001


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

def heat_map(x_group, y_group, x_axis):
    data = np.zeros((5, 4))
    num = np.zeros((5, 4), dtype=int)
    for y in range(5):
        c = y_group[y,:,:]
        for x in range(4):
            with np.errstate(invalid='ignore'):
                mask = c[np.where((x_group >= x_axis[x]) & (x_group <= x_axis[x + 1]))]
                num[y,x] = np.sum(~np.isnan(mask))
                if np.sum(~np.isnan(mask))>=1:
                    data[y, x] = np.nanmedian(mask)
    return(data,num)

def heatmap(ax, num, data, ytick, xtick, ticks,colorbar_tick, **kwargs):
    im1 = ax.imshow(data, origin='lower', **kwargs)
    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5)
    # ax.set_yticks(np.arange(data.shape[0]))
    ax.set_yticks([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5])
    ax.set_xticklabels(xtick)
    ax.set_yticklabels(ytick)
    # cbar = ax.figure.colorbar(im1, ax=ax, ticks=ticks, extend='both', shrink=0.2,aspect=10)
    cbar = ax.figure.colorbar(im1, ax=ax, ticks=ticks, extend='both', shrink=0.6,aspect=10)
    cbar.ax.set_yticklabels(colorbar_tick)

    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center", rotation_mode="anchor")
    for y in range(5):
        for x in range(4):
            ax.text(x, y, num[y, x], ha="center", va="center", color="white")

    return(im1)

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

####  plotting heatmap ########
xticks = [0, 0.8, 1.2, 1.6, 4]
xtick = ['0','0.8','1.2','1.6','>1.6']
yticks = [0,1,2,3,4]
ytick = ['-3 months','-1 months','-8 days','+8 days','+1 months','+3 months']
aridity = read_data(data_path('aridity_ERA5-land_2018-2020-monthly-mean_0d25.npy'))
aridity[aridity < 0] = np.nan
aridity[aridity > 4] = 4

var = [[-0.002, 0.002], [-0.002,0.002],
       [-0.01,0.01],[-0.01,0.01]]

fig = plt.figure(figsize=(3,3), dpi=300, tight_layout=True)
veg = veg_filter(0)
irrigation = read_data(data_path('gmia_v5_aei_pct_720_1440.npy')) # irrigation mask to remove highly irrigated areas
drought = read_data(data_path('Drought_ind_SM123_ERA5-Land.npy'))
var_name_org = ['(a) SIFrel physio (unitless)','(b) LUE (unitless)',
                '(c) Gs ($\mathregular{umol·m^{-2}·s^{-1}}$)',
                '(d)WUE ($\mathregular{umol·s^{-1}}$)']

VD_8d_ano = read_data(data_path('Scope_ano_SIFrel_LUE_Gs_WUE_globe_pft.npy'))
loc = read_data(data_path('Scope_drought_1000pixels.npy'))

########## load observations to remove grid cells if observations have data gaps ##########
Delta = np.zeros((4, 24, 720, 1440)) * np.nan
Delta1 = np.zeros((4, 5, 720, 1440)) * np.nan
var_name1 = ['SIF_rel', 'ET_SSEB_final', 'VOD_ratio']
for tt in range(3):
    SIF_8d_ano = read_data(data_path('test/VDano_physio_' + var_name1[tt] + '_LeaveOut24_LAI.npy'))
    for row in range(720):
        for col in range(1440):
            if ~np.isnan(drought[row, col]) and veg[row, col] >= 0.05 and irrigation[row, col] <= 10:
                drought_ind = drought[row, col]  # 46*8day=368days, 91*8day=730days
                if drought_ind > 12 and drought_ind < 167 - 12:
                    SIF_pixel = SIF_8d_ano[:, row,col]  # 0-137: 2018-03-07 - 2021-02-27; # 0-167: 2018-03-07 - 2021-10-25
                    Delta[tt, :, row, col] = SIF_pixel[int(drought_ind - 12):int(drought_ind + 12)]

for tt in range(3):
    Delta[tt, :, :, :][np.isnan(Delta[0, :, :, :])] = np.nan
    Delta[tt, :, :, :][np.isnan(Delta[1, :, :, :])] = np.nan
    Delta[tt, :, :, :][np.isnan(Delta[2, :, :, :])] = np.nan

    for i, move_window1, move_window2 in zip(range(5), [-12, -4, -1, 1, 4], [-4, -1, 1, 4, 12]):
        Delta1[tt, i, :, :] = np.nanmedian(Delta[tt,int(12 + move_window1):int(12 + move_window2),:,:], axis=0)

    for m in range(5):
        Delta1[tt, m, :, :][np.isnan(Delta1[tt, 0, :, :])] = np.nan
        Delta1[tt, m, :, :][np.isnan(Delta1[tt, 1, :, :])] = np.nan
        Delta1[tt, m, :, :][np.isnan(Delta1[tt, 2, :, :])] = np.nan
        Delta1[tt, m, :, :][np.isnan(Delta1[tt, 3, :, :])] = np.nan
        Delta1[tt, m, :, :][np.isnan(Delta1[tt, 4, :, :])] = np.nan
########## load observations to remove grid cells if observations have data gaps ##########

########## load SIFrel, LUE, Gs, WUE from the SCOPE model
for tt, pic in zip(range(4),range(4)):
    SIF_8d_ano = VD_8d_ano[tt, :, :, :]

    delta = np.zeros((5, 720, 1440)) * np.nan
    for pix in range(1000):
        row = int(loc[pix, 0])
        col = int(loc[pix, 1])
        if np.sum(~np.isnan(SIF_8d_ano[:, row, col]))>0:
            drought_ind = drought[row, col]
            for i, move_window1, move_window2 in zip(range(5), [-12, -4, -1, 1, 4],[-4, -1, 1, 4, 12]):
                SIF_pixel = SIF_8d_ano[:, row, col] # 0-137: 2018-03-07 - 2021-02-27; # 0-167: 2018-03-07 - 2021-10-25
                if drought_ind > 12 and drought_ind < 167 - 12:
                    # print(np.sum(~np.isnan(SIF_pixel[int(drought_ind + move_window1):int(drought_ind + move_window2)])))
                    delta[i,row,col] = np.nanmedian(SIF_pixel[int(drought_ind + move_window1):int(drought_ind + move_window2)])

    for m in range(5):
        delta[m, :, :][np.isnan(Delta1[0, 0, :, :])] = np.nan

    # drawing heatmap
    heatmap_data,num = heat_map(aridity, delta, xticks)
    print(heatmap_data)
    norm = matplotlib.colors.Normalize(vmin=var[tt][0], vmax=var[tt][1])
    ticks = [var[tt][0], 0, var[tt][1]]
    colorbar_tick = [var[tt][0], 0, var[tt][1]]

    ax = fig.add_subplot(2, 2, 1+pic)
    heatmap(ax, num, heatmap_data, ytick, xtick, ticks,colorbar_tick, norm=norm, cmap=plt.get_cmap('coolwarm_r'))
    ax.set_title(var_name_org[tt])
    if pic+1==1 or pic+1==3:
        ax.set_yticklabels(ytick)
        ax.set_ylabel('Drought period')
    else:
        ax.get_yaxis().set_visible(False)

    if pic + 1 == 3 or pic + 1 == 4:
        ax.set_xlabel('Aridity')
    else:
        ax.get_xaxis().set_visible(False)
plt.savefig(data_path('test/Fig5.jpg'),bbox_inches='tight')
