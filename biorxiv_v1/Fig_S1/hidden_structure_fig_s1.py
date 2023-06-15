##########################################################################
# hidden_structure_fig_s1.py
# 
# Author: David Moses (dmoses5@ucmerced.edu)
# 
# Supporting material for "Hidden structure in disordered proteins is adaptive to intracellular changes"
# 
# https://www.biorxiv.org/content/10.1101/2021.11.24.469609v1
##########################################################################

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams["font.weight"] = 'normal'
plt.rcParams["axes.labelweight"] = 'normal'
plt.rc('xtick', labelsize=30) 
plt.rc('ytick', labelsize=30) 
plt.rcParams["xtick.major.size"] = 10
plt.rcParams["ytick.major.size"] = 10
mpl.rcParams['xtick.major.width'] = 4
mpl.rcParams['ytick.major.width'] = 4
mpl.rcParams['xtick.direction'] = 'out'
mpl.rcParams['ytick.direction'] = 'out'
mpl.rcParams['axes.linewidth'] = 4
mpl.rcParams["errorbar.capsize"] = 4
mpl.rc('axes',edgecolor='black')
lines = {'linestyle': 'None', 'linewidth': 3}
plt.rc('lines', **lines)

# Single chromatograms
csvs = ['20211009_SS1_gs0_pbs_chromatogram_for_graphing.csv',
        '20211009_SS2_gs8_pbs_chromatogram_for_graphing.csv',
        '20211009_SS03_gs16_pbs_chromatogram_for_graphing.csv',
        '20211009_SS04_gs24_pbs_chromatogram_for_graphing.csv',
        '20211009_SS05_gs32_pbs_chromatogram_for_graphing.csv',
        '20211009_SS06_gs48_pbs_chromatogram_for_graphing.csv',
        '20211009_SS07_puma_wt_pbs_chromatogram_for_graphing.csv',
        '20211009_SS12_puma_s1_pbs_chromatogram_for_graphing.csv',
        '20211009_SS13_puma_s2_pbs_chromatogram_for_graphing.csv',
        '20211009_SS14_puma_s3_pbs_chromatogram_for_graphing.csv',
        '20211009_SS09_ash1_pbs_chromatogram_for_graphing.csv',
        '20211009_SS10_e1a_pbs_chromatogram_for_graphing.csv',
        '20211009_SS08_p53_pbs_chromatogram_for_graphing.csv',
        '20211009_SS17_fus_pbs_chromatogram_for_graphing.csv']
        
labels = ['GS0', 'GS8', 'GS16', 'GS24', 'GS32', 'GS48', 'WT PUMA',
          'PUMA S1', 'PUMA S2', 'PUMA S3', 'Ash1', 'E1A', 'p53', 'FUS']  

colors = ['dimgrey','grey','darkgrey','silver','lightgray','gainsboro','darkslateblue',
          'blueviolet','violet','darkmagenta','limegreen','teal','royalblue','dodgerblue']  
   
# Create and format the figure
ncols = 4
nrows = 4
wspace=0.4
hspace=0.3
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=False, sharey=False, 
                        figsize=(ncols*10 + (wspace*(ncols-1)), nrows*10 + (hspace*(nrows-1))))
plt.subplots_adjust(wspace=wspace, hspace=hspace)

for i in range(nrows):
    for j in range(ncols):
        index = (ncols * i) + j
        if index < len(csvs):
            df = pd.read_csv(csvs[index])
            axs[i][j].plot(df.Frame, df.Integrated_Intensity, c=colors[index], label=labels[index], linestyle='-')
            axs[i][j].legend(fontsize=30, loc='upper left')
            axs[i][j].xaxis.tick_bottom()
            axs[i][j].yaxis.tick_left()
            axs[i][j].set_xlabel('frame', fontsize=30)
            axs[i][j].set_ylabel('integrated intensity (arb.)', fontsize=30, labelpad=5)
        # else:
        #     axs[i][j].axis('off')
            
# GS chromatograms grouped
csvs = ['20211009_SS1_gs0_pbs_chromatogram_for_graphing.csv',
        '20211009_SS2_gs8_pbs_chromatogram_for_graphing.csv',
        '20211009_SS04_gs24_pbs_chromatogram_for_graphing.csv',
        '20211009_SS05_gs32_pbs_chromatogram_for_graphing.csv',
        '20211009_SS06_gs48_pbs_chromatogram_for_graphing.csv']
labels = ['GS0', 'GS8', 'GS24', 'GS32', 'GS48']
colors = ['dimgrey','grey','silver','lightgray','gainsboro']
for i in range(len(csvs)):
    df = pd.read_csv(csvs[i])
    frame = df.Frame.to_numpy()
    intensity = df.Integrated_Intensity.to_numpy()
    max_intensity = np.amax(intensity)
    min_intensity = np.amin(intensity)
    normalized_intensity = (intensity - min_intensity) / (max_intensity - min_intensity)
    axs[3][2].plot(frame[580:920], normalized_intensity[580:920], c=colors[i], label=labels[i], linestyle='-',zorder=50)
axs[3][2].xaxis.tick_bottom()
axs[3][2].yaxis.tick_left()
axs[3][2].set_xlabel('frame', fontsize=30)
axs[3][2].set_ylabel('normalized intensity', fontsize=30)
axs[3][2].legend(fontsize=30) 

# WT PUMA and scrambles grouped
csvs = ['20211009_SS07_puma_wt_pbs_chromatogram_for_graphing.csv',
        '20211009_SS12_puma_s1_pbs_chromatogram_for_graphing.csv',
        '20211009_SS13_puma_s2_pbs_chromatogram_for_graphing.csv',
        '20211009_SS14_puma_s3_pbs_chromatogram_for_graphing.csv']
labels = ['WT PUMA', 'S1', 'S2', 'S3']
colors = ['darkslateblue','blueviolet','violet','darkmagenta']
for i in range(len(csvs)):
    df = pd.read_csv(csvs[i])
    frame = df.Frame.to_numpy()
    intensity = df.Integrated_Intensity.to_numpy()
    max_intensity = np.amax(intensity)
    min_intensity = np.amin(intensity)
    normalized_intensity = (intensity - min_intensity) / (max_intensity - min_intensity)
    axs[3][3].plot(frame[500:950], normalized_intensity[500:950], c=colors[i], label=labels[i], linestyle='-',zorder=50)
axs[3][3].xaxis.tick_bottom()
axs[3][3].yaxis.tick_left()
axs[3][3].set_xlabel('frame', fontsize=30)
axs[3][3].set_ylabel('normalized intensity', fontsize=30)
axs[3][3].legend(fontsize=30) 

plt.savefig('hidden_structure_fig_s1.png', bbox_inches='tight')
plt.show()
