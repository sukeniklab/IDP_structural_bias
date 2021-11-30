#############################################################################################################################
# hidden_structure_fig_1b_1c_1e.py
# 
# Author: David Moses (dmoses5@ucmerced.edu)
# 
# Supporting material for "Hidden structure in disordered proteins is adaptive to intracellular changes"
# 
# https://www.biorxiv.org/content/10.1101/2021.11.24.469609v1
# 
# Figure 1: Experimental and simulated results for GS-repeat polymers in phosphate buffer
#
# 1A: Cartoons of FRET constructs incorporating GS linkers of different lengths.
#
# 1B: Fluorescence spectra. 1C: Guinier fits. 1D: Simulations: R_e vs. N, R_g vs. N. 1E: R_g vs. R_e.
#
# (1D is done in a different script.)
#
# Dependent on many .csv files, including all referenced in this file and all referenced by hidden_structure_data_processing.
#############################################################################################################################

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.odr.odrpack as odrpack

from hidden_structure_data_processing import get_avg_idr_dirA_intensity_ratio
from hidden_structure_data_processing import preprocess_tq_ng_data
from hidden_structure_data_processing import build_base_correction_factor_df
from hidden_structure_data_processing import get_efret

mpl.style.use('default')
mpl.rcParams['axes.linewidth'] = 7 #set the value globally
mpl.rcParams['xtick.major.size'] = 20
mpl.rcParams['xtick.major.width'] = 7
mpl.rcParams['xtick.minor.size'] = 10
mpl.rcParams['xtick.minor.width'] = 7
mpl.rcParams['ytick.major.size'] = 20
mpl.rcParams['ytick.major.width'] = 7
mpl.rcParams['ytick.labelsize'] = 50
mpl.rcParams['xtick.labelsize'] = 50
mpl.rcParams['ytick.minor.size'] = 10
mpl.rcParams['ytick.minor.width'] = 7
mpl.rcParams['font.size'] = 55
mpl.rcParams['font.sans-serif']='Arial'

# for use with scipy.odr
def f(B, x):
    '''Linear function y = m*x + b'''
    # B is a vector of the parameters.
    # x is an array of the current x values.
    # x is in the same format as the x passed to Data or RealData.
    #
    # Return an array in the same format as y passed to Data or RealData.
    return B[0]*x + B[1]

# Create and format the figure
ncols = 2
nrows = 2
wspace=0
hspace=0
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=False, sharey=False, 
                        figsize=(ncols*10 + (wspace*(ncols-1)), nrows*10 + (hspace*(nrows-1))))
plt.subplots_adjust(wspace=wspace, hspace=hspace)
plt.margins(x=0,y=0)

###############
# Fig 1B
###############

# Read in and preprocess raw data for mTurquoise2 and mNeonGreen base spectra
TQ_data, NG_data = preprocess_tq_ng_data()

# Get the average IDR/dirA intensity ratio for normalization of experimental spectra
avg_idr_dirA_intensity_ratio = get_avg_idr_dirA_intensity_ratio()

# Read in data for GS linkers
GS0_data_1=pd.read_csv("GS0-1-repeat-1.csv",sep=",",skiprows=1).join(pd.read_csv('GS0-2-repeat-1.csv',
                    skiprows=1).iloc[:,3:],rsuffix='2').join(pd.read_csv('GS0-3-repeat-1.csv',
                    skiprows=1).iloc[:,3:],rsuffix='3')
GS0_data_2=pd.read_csv("GS0-1-repeat-2.csv",sep=",",skiprows=1).join(pd.read_csv('GS0-2-repeat-2.csv',
                    skiprows=1).iloc[:,3:],rsuffix='2').join(pd.read_csv('GS0-3-repeat-2.csv',
                    skiprows=1).iloc[:,3:],rsuffix='3')
GS8_data_1=pd.read_csv("GS8-1-repeat-1.csv",sep=",",skiprows=1).join(pd.read_csv('GS8-2-repeat-1.csv',
                    skiprows=1).iloc[:,3:],rsuffix='2').join(pd.read_csv('GS8-3-repeat-1.csv',
                    skiprows=1).iloc[:,3:],rsuffix='3')
GS8_data_2=pd.read_csv("GS8-1-repeat-2.csv",sep=",",skiprows=1).join(pd.read_csv('GS8-2-repeat-2.csv',
                    skiprows=1).iloc[:,3:],rsuffix='2').join(pd.read_csv('GS8-3-repeat-2.csv',
                    skiprows=1).iloc[:,3:],rsuffix='3')
GS16_data_1=pd.read_csv("GS16-1-repeat-1.csv",sep=",",skiprows=1).join(pd.read_csv('GS16-2-repeat-1.csv',
                    skiprows=1).iloc[:,3:],rsuffix='2').join(pd.read_csv('GS16-3-repeat-1.csv',
                    skiprows=1).iloc[:,3:],rsuffix='3')
GS16_data_2=pd.read_csv("GS16-1-repeat-2.csv",sep=",",skiprows=1).join(pd.read_csv('GS16-2-repeat-2.csv',
                    skiprows=1).iloc[:,3:],rsuffix='2').join(pd.read_csv('GS16-3-repeat-2.csv',
                    skiprows=1).iloc[:,3:],rsuffix='3')
GS24_data_1=pd.read_csv("GS24-1-repeat-1.csv",sep=",",skiprows=1).join(pd.read_csv('GS24-2-repeat-1.csv',
                    skiprows=1).iloc[:,3:],rsuffix='2').join(pd.read_csv('GS24-3-repeat-1.csv',
                    skiprows=1).iloc[:,3:],rsuffix='3')
GS24_data_2=pd.read_csv("GS24-1-repeat-2.csv",sep=",",skiprows=1).join(pd.read_csv('GS24-2-repeat-2.csv',
                    skiprows=1).iloc[:,3:],rsuffix='2').join(pd.read_csv('GS24-3-repeat-2.csv',
                    skiprows=1).iloc[:,3:],rsuffix='3')
GS32_data_1=pd.read_csv("GS32-1-repeat-1.csv",sep=",",skiprows=1).join(pd.read_csv('GS32-2-repeat-1.csv',
                    skiprows=1).iloc[:,3:],rsuffix='2').join(pd.read_csv('GS32-3-repeat-1.csv',
                    skiprows=1).iloc[:,3:],rsuffix='3')
GS32_data_2=pd.read_csv("GS32-1-repeat-2.csv",sep=",",skiprows=1).join(pd.read_csv('GS32-2-repeat-2.csv',
                    skiprows=1).iloc[:,3:],rsuffix='2').join(pd.read_csv('GS32-3-repeat-2.csv',
                    skiprows=1).iloc[:,3:],rsuffix='3')
GS48_data_1=pd.read_csv("GS48-1-repeat-1.csv",sep=",",skiprows=1).join(pd.read_csv('GS48-2-repeat-1.csv',
                    skiprows=1).iloc[:,3:],rsuffix='2').join(pd.read_csv('GS48-3-repeat-1.csv',
                    skiprows=1).iloc[:,3:],rsuffix='3')
GS48_data_2=pd.read_csv("GS48-1-repeat-2.csv",sep=",",skiprows=1).join(pd.read_csv('GS48-2-repeat-2.csv',
                    skiprows=1).iloc[:,3:],rsuffix='2').join(pd.read_csv('GS48-3-repeat-2.csv',
                    skiprows=1).iloc[:,3:],rsuffix='3')
                    
# Build a data frame with correction factors for the donor and acceptor at each concentration of each solute
base_corr_fact_df=build_base_correction_factor_df(TQ_data, NG_data)
                    
# Build an E_fret data frame for that IDR
GS0_efret_1=get_efret('GS0',GS0_data_1,range(3,GS0_data_1.shape[1]),base_corr_fact_df,TQ_data,NG_data,True,avg_idr_dirA_intensity_ratio)
GS0_efret_2=get_efret('GS0',GS0_data_2,range(3,GS0_data_2.shape[1]),base_corr_fact_df,TQ_data,NG_data,True,avg_idr_dirA_intensity_ratio)
GS8_efret_1=get_efret('GS8',GS8_data_1,range(3,GS8_data_1.shape[1]),base_corr_fact_df,TQ_data,NG_data,True,avg_idr_dirA_intensity_ratio)
GS8_efret_2=get_efret('GS8',GS8_data_2,range(3,GS8_data_2.shape[1]),base_corr_fact_df,TQ_data,NG_data,True,avg_idr_dirA_intensity_ratio)
GS16_efret_1=get_efret('GS16',GS16_data_1,range(3,GS16_data_1.shape[1]),base_corr_fact_df,TQ_data,NG_data,True,avg_idr_dirA_intensity_ratio)
GS16_efret_2=get_efret('GS16',GS16_data_2,range(3,GS16_data_2.shape[1]),base_corr_fact_df,TQ_data,NG_data,True,avg_idr_dirA_intensity_ratio)
GS24_efret_1=get_efret('GS24',GS24_data_1,range(3,GS24_data_1.shape[1]),base_corr_fact_df,TQ_data,NG_data,True,avg_idr_dirA_intensity_ratio)
GS24_efret_2=get_efret('GS24',GS24_data_2,range(3,GS24_data_2.shape[1]),base_corr_fact_df,TQ_data,NG_data,True,avg_idr_dirA_intensity_ratio)
GS32_efret_1=get_efret('GS32',GS32_data_1,range(3,GS32_data_1.shape[1]),base_corr_fact_df,TQ_data,NG_data,True,avg_idr_dirA_intensity_ratio)
GS32_efret_2=get_efret('GS32',GS32_data_2,range(3,GS32_data_2.shape[1]),base_corr_fact_df,TQ_data,NG_data,True,avg_idr_dirA_intensity_ratio)
GS48_efret_1=get_efret('GS48',GS48_data_1,range(3,GS48_data_1.shape[1]),base_corr_fact_df,TQ_data,NG_data,True,avg_idr_dirA_intensity_ratio)
GS48_efret_2=get_efret('GS48',GS48_data_2,range(3,GS48_data_2.shape[1]),base_corr_fact_df,TQ_data,NG_data,True,avg_idr_dirA_intensity_ratio)

gs0_concat = pd.concat([GS0_efret_1[GS0_efret_1.sol=='Buffer']['fit'].item(),
                        GS0_efret_2[GS0_efret_2.sol=='Buffer']['fit'].item()])
gs8_concat = pd.concat([GS8_efret_1[GS8_efret_1.sol=='Buffer']['fit'].item(),
                        GS8_efret_2[GS8_efret_2.sol=='Buffer']['fit'].item()])
gs16_concat = pd.concat([GS16_efret_1[GS16_efret_1.sol=='Buffer']['fit'].item(),
                         GS16_efret_2[GS16_efret_2.sol=='Buffer']['fit'].item()])
gs24_concat = pd.concat([GS24_efret_1[GS24_efret_1.sol=='Buffer']['fit'].item(),
                        GS24_efret_2[GS24_efret_2.sol=='Buffer']['fit'].item()])
gs32_concat = pd.concat([GS32_efret_1[GS32_efret_1.sol=='Buffer']['fit'].item(),
                        GS32_efret_2[GS32_efret_2.sol=='Buffer']['fit'].item()])
gs48_concat = pd.concat([GS48_efret_1[GS48_efret_1.sol=='Buffer']['fit'].item(),
                        GS48_efret_2[GS48_efret_2.sol=='Buffer']['fit'].item()])

# Wavelengths, f_d, and f_a are the same for all runs of get_efret, so choose any example.
# Average the fit for each GS linker.
df = pd.DataFrame({ "wavelengths":GS16_efret_2[GS16_efret_2.sol=='Buffer']['x_range'].item(),
                    "f_d":GS16_efret_2[GS16_efret_2.sol=='Buffer']['f_d'].item(),
                    "f_a":GS16_efret_2[GS16_efret_2.sol=='Buffer']['f_a'].item(),
                    "fit_GS0":gs0_concat.groupby(gs0_concat.index).mean(),
                    "fit_GS8":gs8_concat.groupby(gs8_concat.index).mean(),
                    "fit_GS16":gs16_concat.groupby(gs16_concat.index).mean(),
                    "fit_GS24":gs24_concat.groupby(gs24_concat.index).mean(),
                    "fit_GS32":gs32_concat.groupby(gs32_concat.index).mean(),
                    "fit_GS48":gs48_concat.groupby(gs48_concat.index).mean() })

colors=['black','dimgrey','grey','darkgrey','silver','lightgray']

axs[0][0].plot(df.wavelengths, 100 * df.fit_GS0 / df.fit_GS0[26], c=colors[0], label='GS0', linewidth=5, zorder=100, ls='-', alpha=1)
axs[0][0].plot(df.wavelengths, 100 * df.fit_GS8 / df.fit_GS8[26], c=colors[1], label='GS8', linewidth=5, zorder=99, ls='-', alpha=0.95)
axs[0][0].plot(df.wavelengths, 100 * df.fit_GS16 / df.fit_GS16[26], c=colors[2], label='GS16', linewidth=5, zorder=98, ls='-', alpha=0.9)
axs[0][0].plot(df.wavelengths, 100 * df.fit_GS24 / df.fit_GS24[26], c=colors[3], label='GS24', linewidth=5, zorder=97, ls='-', alpha=0.85)
axs[0][0].plot(df.wavelengths, 100 * df.fit_GS32 / df.fit_GS32[26], c=colors[4], label='GS32', linewidth=5, zorder=96, ls='-', alpha=0.8)
axs[0][0].plot(df.wavelengths, 100 * df.fit_GS48 / df.fit_GS48[26], c=colors[5], label='GS48', linewidth=5, zorder=95, ls='-', alpha=0.75)
axs[0][0].plot(df.wavelengths, 5000 * df.f_d / df.f_d.sum(), c='cyan', label = 'donor', linewidth=5, ls='-')
axs[0][0].plot(df.wavelengths, 3000 * df.f_a / df.f_a.sum(), c='lime', label = 'acceptor', linewidth=5, ls='-')
axs[0][0].fill_between(df.wavelengths, 5000 * df.f_d / df.f_d.sum(), color='cyan', alpha=0.5)
axs[0][0].fill_between(df.wavelengths, 3000 * df.f_a / df.f_a.sum(), color='lime', alpha=0.3)

axs[0][0].set_xlabel('wavelength (nm)')
axs[0][0].set_ylabel('fluorescence (AU)', labelpad=5)
axs[0][0].legend(fontsize=24, loc='best', frameon=False)
axs[0][0].yaxis.tick_left()
axs[0][0].xaxis.tick_bottom()
axs[0][0].set_xlim([450, 600])

###############
# Fig 1C
###############

# Read in and preprocess raw data for mTurquoise2 and mNeonGreen base spectra
csvs=['20211009_SS1_gs0_pbs_q_I.csv','20211009_SS2_gs8_pbs_q_I.csv','20211009_SS03_gs16_pbs_q_I.csv',
      '20211009_SS04_gs24_pbs_q_I.csv','20211009_SS05_gs32_pbs_q_I.csv','20211009_SS06_gs48_pbs_efa_1_q_I.csv']
colors=['black','dimgrey','grey','darkgrey','silver','lightgray']
labels=['GS0','GS8','GS16','GS24','GS32','GS48']
i0_df=pd.read_csv('20211009_calculated_I0s.csv')

# offset so that all first data points equal first data fit point of gs0

for i in range(len(csvs)):
    df = pd.read_csv(csvs[i])
    # normalize by I(0), i.e., scale so that all I(0) are the same
    scalar = i0_df.I0[0] / i0_df.I0[i]
    axs[0][1].plot(df['Q'][:166], df['I(Q)'][:166] * scalar, c=colors[i], label=labels[i], 
                   alpha=1-(0.05*i), linestyle='solid', linewidth=2, zorder=50)

axs[0][1].set_xscale('log')
axs[0][1].set_xlabel('$q$ (nm$^{-1}$)', labelpad=-5)
axs[0][1].set_ylabel('$I(q)$ (AU)')
axs[0][1].legend(fontsize=32, loc='lower left', frameon=False)
axs[0][1].xaxis.tick_bottom()
axs[0][1].yaxis.tick_left()
axs[0][1].set_xticks([0.01])
axs[0][1].set_xticklabels([0.01])
axs[0][1].set_yticks([0.01, 0.02])
axs[0][1].set_yticklabels([1, 2])

###############
# Fig 1D
###############

axs[1][0].axis('off')

###############
# Fig 1E
###############
data_df = pd.read_csv('hidden_structure_fret_and_saxs_aggregated_data.csv')

linkers = ['GS0','GS8','GS16','GS24','GS32','GS48']
linker_colors = ['black','dimgrey','grey','darkgrey','silver','lightgray']
linker_R_es = np.array([])
linker_R_gs = np.array([])
linker_R_e_errs = np.array([])
linker_R_g_errs = np.array([])
for i in range(len(linkers)):
    R_e = data_df[(data_df.idr == linkers[i]) & (data_df.sol=='Buffer (standard)')].R_e_mean
    R_e_err = data_df[(data_df.idr == linkers[i]) & (data_df.sol=='Buffer (standard)')].R_e_err
    R_g = data_df[(data_df.idr == linkers[i]) & (data_df.sol=='Buffer (standard)')].R_g_mean
    R_g_err = data_df[(data_df.idr == linkers[i]) & (data_df.sol=='Buffer (standard)')].R_g_err
    linker_R_es = np.append(linker_R_es,R_e)
    linker_R_gs = np.append(linker_R_gs,R_g)
    linker_R_e_errs = np.append(linker_R_e_errs,R_e_err)
    linker_R_g_errs = np.append(linker_R_g_errs,R_g_err)
    plt.scatter(R_g,R_e,s=1800,marker='o',c=linker_colors[i],edgecolors='black',linewidth=3,label=linkers[i], zorder=75, alpha=1-(0.05*i))
    plt.errorbar(R_g,R_e,xerr=R_g_err,yerr=R_e_err,capsize=7,capthick=2,ls='none',color='white',elinewidth=3, zorder=100)
# use scipy.odr to draw the linkers regression line and confidence interval
linear = odrpack.Model(f)
# mydata = odrpack.RealData(x, y, sx=sx, sy=sy)
mydata = odrpack.RealData(linker_R_gs, linker_R_es, sx=linker_R_g_errs, sy=linker_R_e_errs)
# instantiate ODR with your data, model and initial parameter estimate
myodr = odrpack.ODR(mydata, linear, beta0=[1., 2.])
myoutput = myodr.run()
# myoutput.pprint()
axs[1][1].set_xlim([31, 45.2])
x_vals = np.array(axs[1][1].get_xlim())
y_vals = myoutput.beta[1] + (myoutput.beta[0] * x_vals)
axs[1][1].plot(x_vals, y_vals, c='cornflowerblue', linestyle='dashed', linewidth=4, zorder=50)
axs[1][1].set_xlabel('$R_g$ ($\AA$)', labelpad=-2)
axs[1][1].set_ylabel('$R^{app}_e$ ($\AA$)', labelpad=10)
axs[1][1].legend(loc='lower right', fontsize=32, handletextpad=0.1, frameon=False)

plt.axis('square')
fig.tight_layout()
plt.savefig('hidden_structure_fig_1b_1c_1e.png', bbox_inches='tight')
plt.show()

