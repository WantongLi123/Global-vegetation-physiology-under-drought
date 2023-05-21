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
import matplotlib.gridspec as gridspec
import random

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


def graph(target, ax, color):
    lon = np.arange(-180,180,0.25)
    lat = np.arange(-60,75,0.25)
    ax.coastlines(linewidth=0.2)
    ax.set_extent([-180, 180, -60, 75], crs=ccrs.PlateCarree())
    cs = ax.pcolor(lon, lat, target[::-1, :], cmap=matplotlib.colors.ListedColormap(color), transform=ccrs.PlateCarree())

    return(cs)

############################## main function

var_name1 = ['LAI','NIRv','SIF','SIF_rel', 'VOD_day','VOD_night','ET_SSEB_final','swvl123', 'VOD_ratio']

var_name2 = ['LAI','NIRv','SIF ($\mathregular{mW/m^{2}/sr/nm}$)','SIFrel (unitless)',
             'VOD (unitless)','VOD (unitless)','ET ($\mathregular{Wm^{-2}}$)','Soil moisture ($\mathregular{m^{3}/m^{3}}$)','VOD ratio (unitless)']
label = ['LAI','NIRv','SIF','SIF_rel', 'Midday VOD','Midnight VOD','ET','Soil Moisture', 'VOD ratio']
color = ['green','purple','olive','darkorange', 'lightblue','grey','brown','black','darkblue']
color2 = ['green','purple','olive','darkorange', 'black','black','brown','black','darkblue']
loc = ['upper left','upper center','upper left','upper center', 'upper right','upper right','upper left','upper center']

veg = veg_filter(0)
irrigation = read_data(data_path('gmia_v5_aei_pct_720_1440.npy')) # irrigation mask to remove highly irrigated areas
t2m_8d = read_data(data_path('t2m_2018-03-07_2021-10-25_0d25_8d.npy'))[:,::-1,:]
SIF_MSC = read_data(data_path('SIF_rel_8d_seasonality.npy'))
drought = read_data(data_path('Drought_ind_SM123_ERA5-Land.npy'))

#####  plotting partioning ########
fig = plt.figure(figsize=(3,4), dpi=300, tight_layout=True)
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.1)
gs1 = gridspec.GridSpecFromSubplotSpec(5, 1, subplot_spec=gs[0], height_ratios=[1,1,1,1,1], hspace=0.5)
gs2 = gridspec.GridSpecFromSubplotSpec(5, 1, subplot_spec=gs[1], height_ratios=[1,1,1,1,1], hspace=0.5)

drought_mask_dry = np.zeros((720, 1440)) * np.nan
drought_mask_wet = np.zeros((720, 1440)) * np.nan
for sub, fi in zip(range(2), [gs1,gs2]):
    aridity = read_data(data_path('aridity_ERA5-land_2018-2020-monthly-mean_0d25.npy'))
    aridity[aridity < 0] = np.nan
    aridity[aridity > 4] = 4
    aridity1 = np.repeat(aridity[np.newaxis, :, :], 25, axis=0)
    aridity2 = np.repeat(aridity1[np.newaxis, :, :, :], 9, axis=0)

    SIF_ano_180d = np.zeros((9,167,720,1440)) * np.nan
    for tt in [0,1,2,3,4,5,6,8]:
        SIF_ano_180d[tt,:,:,:] = read_data(data_path(var_name1[tt] + '_8d_ano_globeDrought.npy'))

    SIF_ano_180d[7,:,:,:] = read_data(data_path('swvl123_2018-03-07_2021-10-25_0d25_8d.npy'))[:,::-1,:]


    for tt in range(9):
        SIF_ano_180d[tt,:,:,:][np.isnan(SIF_ano_180d[0, :, :, :])] = np.nan
        SIF_ano_180d[tt,:,:,:][np.isnan(SIF_ano_180d[1, :, :, :])] = np.nan
        SIF_ano_180d[tt,:,:,:][np.isnan(SIF_ano_180d[2, :, :, :])] = np.nan
        SIF_ano_180d[tt,:,:,:][np.isnan(SIF_ano_180d[3, :, :, :])] = np.nan
        SIF_ano_180d[tt,:,:,:][np.isnan(SIF_ano_180d[4, :, :, :])] = np.nan
        SIF_ano_180d[tt,:,:,:][np.isnan(SIF_ano_180d[5, :, :, :])] = np.nan
        SIF_ano_180d[tt,:,:,:][np.isnan(SIF_ano_180d[6, :, :, :])] = np.nan
        SIF_ano_180d[tt,:,:,:][np.isnan(SIF_ano_180d[7, :, :, :])] = np.nan
        SIF_ano_180d[tt,:,:,:][np.isnan(SIF_ano_180d[8, :, :, :])] = np.nan
        SIF_ano_180d[tt, :, :, :][t2m_8d <= 5 + 273.15] = np.nan  # T<5˚C winter season
        SIF_ano_180d[tt, :, :, :][SIF_MSC <= 0.2] = np.nan  # sif<0.2 non-growing season

    ax1 = fig.add_subplot(fi[1])
    ax2 = fig.add_subplot(fi[2])
    ax3 = fig.add_subplot(fi[3])
    ax4 = fig.add_subplot(fi[4])
    ax5 = fig.add_subplot(fi[0], projection=ccrs.PlateCarree())

    # SIF anomalies during 180 days drought
    output = np.zeros((9, 25, 720, 1440)) * np.nan
    for row in range(720):
        for col in range(1440):
            if ~np.isnan(drought[row, col]) and veg[row, col] > 0.05 and irrigation[row, col] < 10:
                drought_ind = drought[row, col]
                if drought_ind >= 12 and drought_ind <= 167 - 13:
                    drought_mask_dry[row, col] = 1
                    drought_mask_wet[row, col] = 1

                    output[:, :, row, col] = SIF_ano_180d[:, int(drought_ind - 12):int(drought_ind + 13), row, col]  # 0,1,2,3,...,12,...,24

                    for tt in range(9):
                        # Grid cells are only considered if data are available for at least 20 out of the 24 time steps during drought
                        if np.sum(~np.isnan(output[0, :, row, col])) < 20 or np.sum(~np.isnan(output[1, :, row, col])) < 20 \
                                or np.sum(~np.isnan(output[2, :, row, col])) < 20 or np.sum(~np.isnan(output[3, :, row, col])) < 20 \
                                or np.sum(~np.isnan(output[4, :, row, col])) < 20 or np.sum(~np.isnan(output[5, :, row, col])) < 20 \
                                or np.sum(~np.isnan(output[6, :, row, col])) < 20 or np.sum(~np.isnan(output[7, :, row, col])) < 20:
                            output[tt, :, row, col] = np.nan

    ax = ax5
    if sub == 0:
        for row in range(720):
            for col in range(1440):
                if aridity[row, col] > 1 and np.sum(~np.isnan(output[0, :, row, col])) > 0:
                    drought_mask_dry[row, col] = 2
                else:
                    output[:, :, row, col] = np.nan
        graph(drought_mask_dry[60:600, :], ax, ['white','red']) # global plot of dry regions
        ax.set_yticks([-60, -30, 0, 30, 60])
        ax.set_yticklabels(['60˚S', '30˚S', '0˚', '30˚N', '60˚N'])

        n = np.sum(~np.isnan(np.nanmean(output[0, :, :, :], axis=0)))
        ax.set_title('(a) Dry: N=' + str(n))
        print('draw:',np.sum(drought_mask_dry==2))

    if sub == 1:
        for row in range(720):
            for col in range(1440):
                if aridity[row, col] <= 1 and np.sum(~np.isnan(output[0, :, row, col])) > 0:
                    drought_mask_wet[row, col] = 2
                else:
                    output[:, :, row, col] = np.nan
        graph(drought_mask_wet[60:600, :], ax, ['white','red']) # global plot of wet regions
        ax.get_yaxis().set_visible(False)

        n = np.sum(~np.isnan(np.nanmean(output[0, :, :, :], axis=0)))
        ax.set_title('(b) Wet: N=' + str(n))
        print('draw:',np.sum(drought_mask_wet==2))

    title = [['(c)', '(e)', '(g)', '(i)'], ['(d)', '(f)', '(h)', '(j)']]

    for tt in range(9):
        if tt==0:
            ax=ax1
            ax.set_title(title[sub][0])
            ax.get_xaxis().set_visible(False)
            if sub == 1:
                ax.get_yaxis().set_visible(False)
        if tt==1:
            ax = ax1.twinx()
            if sub == 0:
                ax.get_yaxis().set_visible(False)

        if tt==2:
            ax=ax2
            ax.set_title(title[sub][1])
            ax.get_xaxis().set_visible(False)
            if sub == 1:
                ax.get_yaxis().set_visible(False)
        if tt==3:
            ax = ax2.twinx()
            if sub == 0:
                ax.get_yaxis().set_visible(False)
        if tt == 4:
            ax=ax3
            ax.set_title(title[sub][2])
            ax.get_xaxis().set_visible(False)
            if sub == 1:
                ax.get_yaxis().set_visible(False)
        if tt==5:
            ax = ax3

        if tt == 6:
            ax = ax4
            ax.set_title(title[sub][3])
            ax.set_xlim([0,24])
            ax.set_xticks([0,8,11,13,16,24])
            ax.set_xticklabels(['-3M', '-1M', '-8D', '+8D', '+1M','+3M'])
            if sub == 1:
                ax.get_yaxis().set_visible(False)
        if tt == 7:
            ax = ax4.twinx()
            if sub == 0:
                ax.get_yaxis().set_visible(False)

        if tt == 8:
            ax = ax3.twinx()
            if sub == 0:
                ax.get_yaxis().set_visible(False)


        y11 = np.nanmean(output[tt,:,:,:], axis=1)
        y1 = np.nanmean(y11, axis=1)
        time = np.arange(25)
        ax.plot(time, y1, c=color[tt], linewidth=0.3, label=label[tt])
        ax.set_ylabel(var_name2[tt])


        ######################## standard error
        subset = np.zeros((25,720,1440)) * np.nan
        subset1 = np.zeros((9,25,720,1440)) * np.nan

        for row in range(240):
            for col in range(480):
                subset1[0,:, row*3, col*3] = output[tt, :, row*3, col*3]
                subset1[1,:, row*3+1, col*3] = output[tt, :, row*3+1, col*3]
                subset1[2,:, row*3+2, col*3] = output[tt, :, row*3+2, col*3]
                subset1[3,:, row*3, col*3+1] = output[tt, :, row*3, col*3+1]
                subset1[4,:, row*3+1, col*3+1] = output[tt, :, row*3+1, col*3+1]
                subset1[5,:, row*3+2, col*3+1] = output[tt, :, row*3+2, col*3+1]
                subset1[6,:, row*3, col*3+2] = output[tt, :, row*3, col*3+2]
                subset1[7,:, row*3+1, col*3+2] = output[tt, :, row*3+1, col*3+2]
                subset1[8,:, row*3+2, col*3+2] = output[tt, :, row*3+2, col*3+2]

        y_std1 = np.zeros((9,25))*np.nan
        for pix in range(9):
            mean = np.nanmean(subset1[pix,:,:,:], axis=0)
            num = np.sum(~np.isnan(mean))
            y_std11 = subset1[pix,:,:,:].reshape(25, -1)
            y_std1[pix, :] = np.nanstd(y_std11, axis=1) / np.sqrt(num)

        y_std = np.nanmean(y_std1, axis=0)
        print(np.sqrt(num), y_std)
        ax.fill_between(time, y1 + y_std, y1 - y_std, facecolor=color[tt], alpha=0.2)
        ####################### standard error


        if tt!=7 or tt!=4 or tt!=5:
            # ax.axhline(0, linestyle='--', c='grey')
            yabs_max = abs(max(ax.get_ylim(), key=abs))
            ax.set_ylim(ymin=-yabs_max, ymax=yabs_max)
        if tt==7:
            print(1)
            ax.set_ylim(ymin=0.15, ymax=0.45)
            # ax.set_ylim(ymin=0, ymax=0.45)
            # ax.set_ylim(ymin=3, ymax=8)
        if tt==4 or tt==5:
            # ax.axhline(0, linestyle='--', c='grey')
            ax.set_ylim(ymin=-0.05, ymax=0.05)
            # ax.set_ylim(ymin=-0.1, ymax=0.1)
            # ax.set_ylim(ymin=-0.1, ymax=0.1)
            if sub==0:
                ax.legend(loc=loc[tt], frameon=False)

        ax.tick_params(axis='y', colors=color2[tt])
        ax.yaxis.label.set_color(color2[tt])


plt.savefig(data_path('test/Fig2.jpg'),bbox_inches='tight')
