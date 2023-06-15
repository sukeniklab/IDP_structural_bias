########################################################################################################
# hidden_structure_fig_s2.py
# 
# Author: David Moses (dmoses5@ucmerced.edu)
# 
# Supporting material for "Hidden structure in disordered proteins is adaptive to intracellular changes"
# 
# https://www.biorxiv.org/content/10.1101/2021.11.24.469609v1
########################################################################################################

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from math import sqrt
from math import e

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

# Read in Kratky plot data
csvs = ['20211009_SS1_gs0_pbs_dimensionless_kratky.csv',
        '20211009_SS2_gs8_pbs_dimensionless_kratky.csv',
        '20211009_SS03_gs16_pbs_dimensionless_kratky.csv',
        '20211009_SS04_gs24_pbs_dimensionless_kratky.csv',
        '20211009_SS05_gs32_pbs_dimensionless_kratky.csv',
        '20211009_SS06_gs48_pbs_efa_1_dimensionless_kratky.csv',
        '20211009_SS07_puma_wt_pbs_efa_1_dimensionless_kratky.csv',
        '20211009_SS12_puma_s1_pbs_dimensionless_kratky.csv',
        '20211009_SS13_puma_s2_pbs_dimensionless_kratky.csv',
        '20211009_SS14_puma_s3_pbs_dimensionless_kratky.csv',
        '20211009_SS09_ash1_pbs_dimensionless_kratky.csv',
        '20211009_SS10_e1a_pbs_efa_3_SV_1_dimensionless_kratky.csv',
        '20211009_SS08_p53_pbs_dimensionless_kratky.csv',
        '20211009_SS17_fus_pbs_efa_0_dimensionless_kratky.csv']
        
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
            df_trunc = df[df.qRg < 9]
            axs[i][j].plot(df_trunc['qRg'], df_trunc['(qRg)^2*I(q)/I(0)'], c=colors[index], label=labels[index], linestyle='-')
            axs[i][j].legend(fontsize=30, loc='lower center')
            axs[i][j].xaxis.tick_bottom()
            axs[i][j].yaxis.tick_left()
            axs[i][j].set_xlabel('$qR_g$', fontsize=30)
            axs[i][j].set_ylabel('$(qR_g)^2*I(q)/I(0)$', fontsize=30)
            axs[i][j].axhline(3/e, color='gray', lw=1, alpha=1, linestyle='dashed')
            axs[i][j].axvline(sqrt(3), color='gray', lw=1, alpha=1, linestyle='dashed')
        else:
            axs[i][j].axis('off')

plt.savefig('hidden_structure_fig_s2.png', bbox_inches='tight')
plt.show()
