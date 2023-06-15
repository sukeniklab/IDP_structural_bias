##########################################################################################################
# hidden_structure_fig_s4.py
#
# Author: David Moses (dmoses5@ucmerced.edu)
# 
# Supporting material for "Hidden structure in disordered proteins is adaptive to intracellular changes"
# 
# https://www.biorxiv.org/content/10.1101/2021.11.24.469609v1
##########################################################################################################

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams["font.weight"] = 'normal'
plt.rcParams["axes.labelweight"] = 'normal'
plt.rc('xtick', labelsize=24) 
plt.rc('ytick', labelsize=24) 
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

# Read in Guinier plots for GS linkers to interpolate or extrapolate GS equivalents
gs0_df = pd.read_csv('20211009_SS1_gs0_pbs_guinier.csv')
gs8_df = pd.read_csv('20211009_SS2_gs8_pbs_guinier.csv')
gs16_df = pd.read_csv('20211009_SS03_gs16_pbs_guinier.csv')
gs24_df = pd.read_csv('20211009_SS04_gs24_pbs_guinier.csv')
gs32_df = pd.read_csv('20211009_SS05_gs32_pbs_guinier.csv')
gs48_df = pd.read_csv('20211009_SS06_gs48_pbs_efa_1_guinier.csv')

# for use with curve_fit
def linearFunc(x,intercept,slope):
    y = intercept + slope * x
    return y

# use q**2_data and ln(I(q))_data positions 3 and 102.
def get_guinier_fit_slope(df):
    return ((df['ln(I(q))_fit'][40] - df['ln(I(q))_fit'][3]) / (df['q**2_fit'][40] - df['q**2_fit'][3]))

def get_guinier_fit_y_intercept(df):
    # y = mx + b, so b = y - mx
    return (df['ln(I(q))_fit'][40] - (get_guinier_fit_slope(df) * df['q**2_fit'][40]))

linker_Ns = [0, 16, 32, 48, 64, 96]
guinier_fit_slopes = [get_guinier_fit_slope(gs0_df), get_guinier_fit_slope(gs8_df), get_guinier_fit_slope(gs16_df),
                      get_guinier_fit_slope(gs24_df), get_guinier_fit_slope(gs32_df), get_guinier_fit_slope(gs48_df)]
a_fit,cov=curve_fit(linearFunc,linker_Ns,guinier_fit_slopes)
gs_guinier_slope_regression_intercept = a_fit[0]
gs_guinier_slope_regression_slope = a_fit[1]

# Read in guinier plots and linear fits
csvs = ['20211009_SS1_gs0_pbs_guinier.csv',
        '20211009_SS2_gs8_pbs_guinier.csv',
        '20211009_SS03_gs16_pbs_guinier.csv',
        '20211009_SS04_gs24_pbs_guinier.csv',
        '20211009_SS05_gs32_pbs_guinier.csv',
        '20211009_SS06_gs48_pbs_efa_1_guinier.csv',
        '20211009_SS07_puma_wt_pbs_efa_1_guinier.csv',
        '20211009_SS12_puma_s1_pbs_guinier.csv',
        '20211009_SS13_puma_s2_pbs_guinier.csv',
        '20211009_SS14_puma_s3_pbs_guinier.csv',
        '20211009_SS09_ash1_pbs_guinier.csv',
        '20211009_SS10_e1a_pbs_efa_3_SV_1_guinier.csv',
        '20211009_SS08_p53_pbs_guinier.csv',
        '20211009_SS17_fus_pbs_efa_0_guinier.csv']
        
labels = ['GS0', 'GS8', 'GS16', 'GS24', 'GS32', 'GS48', 'WT PUMA', 'PUMA S1', 
          'PUMA S2', 'PUMA S3', 'Ash1', 'E1A', 'p53', 'FUS']  
colors = ['dimgrey','grey','darkgrey','silver','lightgray','gainsboro','darkslateblue',
          'blueviolet','violet','darkmagenta','limegreen','teal','royalblue','dodgerblue']
Ns = [0, 16, 32, 48, 64, 96, 34, 34, 34, 34, 83, 40, 61, 163]   

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
            if labels[index] not in ['GS0', 'GS8', 'GS16', 'GS24', 'GS32', 'GS48']:
                # Compute slope of implied Guinier fit line of GS-equivalent of current IDR
                gs_equiv_guinier_fit_slope = gs_guinier_slope_regression_intercept + (gs_guinier_slope_regression_slope * Ns[index])
                # Start GS-equivalent line from X=0
                gs_equiv_start_X = 0
                # X end point of GS-equivalent line is the same as last X for current IDR fit line
                gs_equiv_end_X = df['q**2_fit'][df['q**2_fit'].last_valid_index()]
                # Set y-intercept of GS-equivalent line to be the same as y-intercept of GS0 fit line 
                gs_equiv_start_Y = get_guinier_fit_y_intercept(gs0_df)
                # Y end point of GS-equivalent line depends on the GS-equivalent slope
                gs_equiv_end_Y = gs_equiv_start_Y + (gs_equiv_guinier_fit_slope * (gs_equiv_end_X - gs_equiv_start_X))
            # Adjust plots for IDR so that IDR fit line has same y-intercept as GS0 fit line
            y_int_offset = get_guinier_fit_y_intercept(gs0_df) - get_guinier_fit_y_intercept(df)
            # Plot the lines of the IDR (with offset) and the GS-equivalent
            axs[i][j].scatter(df['q**2_data'][5:], df['ln(I(q))_data'][5:] + y_int_offset, c=colors[index], alpha=0.5, zorder=100)
            axs[i][j].plot(df['q**2_fit'], df['ln(I(q))_fit'] + y_int_offset, c=colors[index], label=labels[index], linestyle='-', linewidth=4, zorder=75)
            if labels[index] not in ['GS0', 'GS8', 'GS16', 'GS24', 'GS32', 'GS48']:
                axs[i][j].plot([gs_equiv_start_X, gs_equiv_end_X], [gs_equiv_start_Y, gs_equiv_end_Y], c='black', 
                               label=labels[index] + ' GS-equiv (N=' + str(Ns[index]) + ')', 
                               linestyle='dotted', linewidth=2, zorder=50)
            axs[i][j].legend(fontsize=22, loc='upper right')
            axs[i][j].xaxis.tick_bottom()
            axs[i][j].yaxis.tick_left()
            axs[i][j].set_xlabel('$q^2$', fontsize=36)
            axs[i][j].set_ylabel('$ln(I(q))$', fontsize=36, labelpad=1)
            if labels[index] == 'FUS':
                axs[i][j].set_xticks([0.0000, 0.0002, 0.0004, 0.0006])
        else:
            axs[i][j].axis('off')
plt.tight_layout()
plt.savefig('hidden_structure_fig_s4.png', bbox_inches='tight')
plt.show()
