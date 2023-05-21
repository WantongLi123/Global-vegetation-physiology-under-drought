#!/usr/bin/env python
import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import sys
import scipy
from sklearn.linear_model import LinearRegression, TheilSenRegressor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pymannkendall as mk
from sklearn.ensemble import RandomForestRegressor
import shap
from matplotlib import rc
import pandas as pd
import xarray as xr
from matplotlib import colors

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREAD"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

rc('font',**{'family':'sans-serif','sans-serif':['Arial']})
plt.rcParams.update({'font.size': 5})
plt.rcParams['axes.unicode_minus']=False
plt.rcParams['lines.markersize'] = 0.2

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

def relative_imp(n, data, mutli_vari,target_name):
    data = data.reshape(n, -1)
    data = np.transpose(data)

    test_data = data[~np.all(data == 0, axis=1)]
    test_data = test_data[~np.any(np.isnan(test_data), axis=1)]
    test_data = test_data[~np.any(np.isinf(test_data), axis=1)]
    for v in range(n):
        print(np.nanmin(test_data[:,v]),np.nanmax(test_data[:,v]))

    print('length of data:', len(test_data[:, 0]))

    importance_mean = np.nan
    if len(test_data[:, 0])>100:
        df_test = pd.DataFrame({mutli_vari[v]: test_data[:, v] for v in range(n)})
        X_test, y_test = df_test.drop(target_name, axis=1), df_test[target_name]
        print(X_test.head(100))

        rf = RandomForestRegressor(n_estimators=100,
                                       max_features=0.3,
                                       n_jobs=8,
                                       bootstrap=True,
                                       oob_score=True,
                                       random_state=42)

        rf.fit(X_test, y_test)
        print(rf.oob_score_)

        if rf.oob_score_ > 0:
            explainer = shap.TreeExplainer(rf)
            shap_values = explainer.shap_values(X_test)
            # shap_values = explainer.shap_values(X_test.head(100))
            print(np.shape(shap_values))

            shap_values = abs(shap_values)
            importance_mean = np.nanmean(shap_values, axis=0)  # Mean[abs(positive and negative SHAP)]---->importance
            print(importance_mean)
    return(importance_mean)

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


if __name__ == '__main__':
    log_string = 'data-processing :'

# take care: each SHAP importance calculation will take over 1 day
# ########### for SIF_rel ##########################################
#     tar_names = ['SIF_rel physio', 'ET_SSEB8 physio', 'VOD_ratio physio']
#     target_name = tar_names[0]
#     mutli_vari_1 = ['SIF_rel physio', 'Drought-development TP', 'Drought-development TEM', 'Drought-development SW',
#                     'Drought-development VPD', 'Drought-development SM', 'Aridity', 'Tree:Non-tree', 'Drought-development duration']
#     mutli_vari_2 = ['SIF_rel physio', 'Drought-recovery TP', 'Drought-recovery TEM', 'Drought-recovery SW',
#                     'Drought-recovery VPD', 'Drought-recovery SM', 'Drought-development TP', 'Drought-development TEM',
#                     'Drought-development SW', 'Drought-development VPD', 'Drought-development SM', 'Aridity', 'Tree:Non-tree',
#                     'Drought-development duration', 'Drought-recovery duration']
#     # run SHAP importance
#     # data1 = read_data(data_path('drought_before_variables_SIF_rel.npy'))
#     data2 = read_data(data_path('drought_after_variables_SIF_rel.npy'))
#     # Importance_mean1 = relative_imp(9, data1, mutli_vari_1,target_name)
#     Importance_mean2 = relative_imp(15, data2, mutli_vari_2,target_name)
#     # np.save(data_path('rtest/drought_before_Importance_SIF_rel'), Importance_mean1)
#     np.save(data_path('test/drought_after_Importance_SIF_rel'), Importance_mean2)
# ##### OOB: 0.40; 0.41


# ######## for ET ##########################################
#     tar_names = ['SIF_rel physio', 'ET physio', 'VOD_ratio physio']
#     target_name = tar_names[1]
#     mutli_vari_1 = ['ET physio', 'Drought-development TP', 'Drought-development TEM', 'Drought-development SW',
#                     'Drought-development VPD', 'Drought-development SM', 'Aridity', 'Tree:Non-tree',
#                     'Drought-development duration']
#     mutli_vari_2 = ['ET physio', 'Drought-recovery TP', 'Drought-recovery TEM', 'Drought-recovery SW',
#                     'Drought-recovery VPD', 'Drought-recovery SM', 'Drought-development TP', 'Drought-development TEM',
#                     'Drought-development SW', 'Drought-development VPD', 'Drought-development SM', 'Aridity',
#                     'Tree:Non-tree',
#                     'Drought-development duration', 'Drought-recovery duration']
#     # run SHAP importance
#     # data1 = read_data(data_path('drought_before_variables_ET_SSEB_final.npy'))
#     data2 = read_data(data_path('drought_after_variables_ET_SSEB_final.npy'))
#     # Importance_mean1 = relative_imp(9, data1, mutli_vari_1,target_name)
#     Importance_mean2 = relative_imp(15, data2, mutli_vari_2,target_name)
#     # np.save(data_path('test/drought_before_Importance_ET_SSEB_final'), Importance_mean1)
#     np.save(data_path('test/drought_after_Importance_ET_SSEB_final'), Importance_mean2)
#
# ######## OOB: 0.45; 0.38



# ######## for VODratio ##########################################
#     tar_names = ['SIF_rel physio', 'ET_SSEB8 physio', 'VOD_ratio physio']
#     target_name = tar_names[2]
#     mutli_vari_1 = ['VOD_ratio physio', 'Drought-development TP', 'Drought-development TEM', 'Drought-development SW',
#                     'Drought-development VPD', 'Drought-development SM', 'Aridity', 'Tree:Non-tree',
#                     'Drought-development duration']
#     mutli_vari_2 = ['VOD_ratio physio', 'Drought-recovery TP', 'Drought-recovery TEM', 'Drought-recovery SW',
#                     'Drought-recovery VPD', 'Drought-recovery SM', 'Drought-development TP', 'Drought-development TEM',
#                     'Drought-development SW', 'Drought-development VPD', 'Drought-development SM', 'Aridity',
#                     'Tree:Non-tree',
#                     'Drought-development duration', 'Drought-recovery duration']
#     # run SHAP importance
#     # data1 = read_data(data_path('drought_before_variables_VOD_ratio.npy'))
#     data2 = read_data(data_path('drought_after_variables_VOD_ratio.npy'))
#     # Importance_mean1 = relative_imp(9, data1, mutli_vari_1,target_name)
#     Importance_mean2 = relative_imp(15, data2, mutli_vari_2,target_name)
#     # np.save(data_path('test/drought_before_Importance_VOD_ratio'), Importance_mean1)
#     np.save(data_path('test/drought_after_Importance_VOD_ratio'), Importance_mean2)
#
# ######## OOB: 0.53; 0.43


########## plot pre-drought ###############
    Importance1 = read_data(data_path('test/drought_before_Importance_SIF_rel.npy'))
    Importance3 = read_data(data_path('test/drought_before_Importance_ET_SSEB_final.npy'))
    Importance5 = read_data(data_path('test/drought_before_Importance_VOD_ratio.npy'))

    mutli_vari_1 = ['Precipitation', 'Temperature', 'Radiation',
                    'VPD', 'Soil moisture', 'Aridity', 'Tree/(Grass+Shrub)',
                    '(Dev.) Duration']

    edgecolor1 = ['b', 'b', 'b', 'b', 'b', 'brown', 'brown', 'b']

    sig1 = [['','','*','*','','*','','*'],['*','*','*','','','*','*',''],['','','','*','','*','*','*']]

    df1 = pd.DataFrame({
        'Group': mutli_vari_1,
        'Value': Importance1,
        'Color': edgecolor1,
        'sign': sig1[0]
    })
    df1 = df1.sort_values(by=['Value']).reset_index()
    print(df1)

    df3 = pd.DataFrame({
        'Group': mutli_vari_1,
        'Value': Importance3,
        'Color': edgecolor1,
        'sign': sig1[1]
    })
    df3 = df3.sort_values(by=['Value']).reset_index()

    df5 = pd.DataFrame({
        'Group': mutli_vari_1,
        'Value': Importance5,
        'Color': edgecolor1,
        'sign': sig1[2]
    })
    df5 = df5.sort_values(by=['Value']).reset_index()

    fig = plt.figure(figsize=(4, 4), dpi=300, tight_layout=True)
    ax = fig.add_subplot(2, 3, 1)
    ax.barh(y=df1.Group, width=df1.Value, color='lightgrey', edgecolor=df1.Color)
    # ax.set_xlabel('SHAP Importance for SIFrel physio')
    ax.set_title('(a)')
    ax.set_xlim(0,0.01)

    # add sign to indicate same with correlation result
    for i, v in enumerate(df1.Value):
        ax.text(v, i, df1.sign[i], color='black', fontsize=6)

    ax = fig.add_subplot(2, 3, 2)
    ax.barh(y=df3.Group, width=df3.Value, color='lightgrey', edgecolor=df3.Color)
    # ax.set_xlabel('SHAP Importance for ET physio')
    ax.set_title('(b)')
    ax.set_xlim(0,1)

    # add sign to indicate same with correlation result
    for i, v in enumerate(df3.Value):
        ax.text(v, i, df3.sign[i], color='black', fontsize=6)

    ax = fig.add_subplot(2, 3, 3)
    ax.barh(y=df5.Group, width=df5.Value, color='lightgrey', edgecolor=df5.Color)
    # ax.set_xlabel('SHAP Importance for VOD ratio physio')
    ax.set_title('(c)')
    ax.set_xlim(0,0.005)

    # add sign to indicate same with correlation result
    for i, v in enumerate(df5.Value):
        ax.text(v, i, df5.sign[i], color='black', fontsize=6)

    ########### plot post-drought ###############
    Importance2 = read_data(data_path('test/drought_after_Importance_SIF_rel.npy'))
    Importance4 = read_data(data_path('test/drought_after_Importance_ET_SSEB_final.npy'))
    Importance6 = read_data(data_path('test/drought_after_Importance_VOD_ratio.npy'))
    mutli_vari_2 = ['Precipitation', 'Temperature', 'Radiation',
                    'VPD', 'Soil moisture', '(Dev.) Precipitation', '(Dev.) Temperature',
                    '(Dev.) Radiation', '(Dev.) VPD', '(Dev.) Soil moisture', 'Aridity',
                    'Tree/(Grass+Shrub)',
                    '(Dev.) Duration', '(Recov.) Duration']
    edgecolor2 = ['b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'brown', 'brown', 'b', 'b']
    sig2 = [['*','','','*','*','','','','','','','','',''],
            ['','','','*','*','','','','','','','','',''],
            ['','','','*','','','','','','','*','','','']]

    df2 = pd.DataFrame({
        'Group': mutli_vari_2,
        'Value': Importance2,
        'Color': edgecolor2,
        'sign': sig2[0]
    })
    df2 = df2.sort_values(by=['Value']).reset_index()

    df4 = pd.DataFrame({
        'Group': mutli_vari_2,
        'Value': Importance4,
        'Color': edgecolor2,
        'sign': sig2[1]
    })
    df4 = df4.sort_values(by=['Value']).reset_index()

    df6 = pd.DataFrame({
        'Group': mutli_vari_2,
        'Value': Importance6,
        'Color': edgecolor2,
        'sign': sig2[2]
    })
    df6 = df6.sort_values(by=['Value']).reset_index()

    # fig = plt.figure(figsize=(4, 2), dpi=300, tight_layout=True)
    ax = fig.add_subplot(2, 3, 4)
    ax.barh(y=df2.Group, width=df2.Value, color='lightgrey', edgecolor=df2.Color)
    ax.set_xlabel('SHAP Importance for SIFrel physio')
    ax.set_title('(d)')
    ax.set_xlim(0, 0.015)
    # add sign to indicate same with correlation result
    for i, v in enumerate(df2.Value):
        ax.text(v, i-0.2, df2.sign[i], color='black', fontsize=6)

    ax = fig.add_subplot(2, 3, 5)
    ax.barh(y=df4.Group, width=df4.Value, color='lightgrey', edgecolor=df4.Color)
    ax.set_xlabel('SHAP Importance for ET physio')
    ax.set_title('(e)')
    ax.set_xlim(0,1)
    # add sign to indicate same with correlation result
    for i, v in enumerate(df4.Value):
        ax.text(v, i-0.2, df4.sign[i], color='black', fontsize=6)

    ax = fig.add_subplot(2, 3, 6)
    ax.barh(y=df6.Group, width=df6.Value, color='lightgrey', edgecolor=df6.Color)
    ax.set_xlabel('SHAP Importance for VOD ratio physio')
    ax.set_title('(f)')
    ax.set_xlim(0,0.002)
    # add sign to indicate same with correlation result
    for i, v in enumerate(df6.Value):
        ax.text(v, i-0.2, df6.sign[i], color='black', fontsize=6)

plt.savefig(data_path('test/Fig4.jpg'),bbox_inches='tight')

