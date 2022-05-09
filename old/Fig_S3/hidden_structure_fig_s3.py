#########################################################################################################
# hidden_structure_fig_s3.py
#
# Author: David Moses (dmoses5@ucmerced.edu)
# 
# Supporting material for "Hidden structure in disordered proteins is adaptive to intracellular changes"
# 
# https://www.biorxiv.org/content/10.1101/2021.11.24.469609v1
######################################################################################################### 

from hidden_structure_data_processing import get_avg_idr_dirA_intensity_ratio
from hidden_structure_data_processing import preprocess_tq_ng_data
from hidden_structure_data_processing import build_base_correction_factor_df
from hidden_structure_data_processing import get_efret

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams["font.weight"] = 'normal'
plt.rcParams["axes.labelweight"] = 'normal'
plt.rc('xtick', labelsize=36) 
plt.rc('ytick', labelsize=36) 
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

# for use with curve_fit
def linearFunc(x,intercept,slope):
    y = intercept + slope * x
    return y

# Read in and preprocess raw data for mTurquoise2 and mNeonGreen base spectra
TQ_data, NG_data = preprocess_tq_ng_data()

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
PUMA_data_1=pd.read_csv("puma-wt-1-repeat-1.csv",sep=",",skiprows=1).join(pd.read_csv('puma-wt-2-repeat-1.csv',
                    skiprows=1).iloc[:,3:],rsuffix='2').join(pd.read_csv('puma-wt-3-repeat-1.csv',
                    skiprows=1).iloc[:,3:],rsuffix='3')
PUMA_data_2=pd.read_csv("puma-wt-1-repeat-2.csv",sep=",",skiprows=1).join(pd.read_csv('puma-wt-2-repeat-2.csv',
                    skiprows=1).iloc[:,3:],rsuffix='2').join(pd.read_csv('puma-wt-3-repeat-2.csv',
                    skiprows=1).iloc[:,3:],rsuffix='3')
PUMAS1_data_1=pd.read_csv("puma-s1-1-repeat-1.csv",sep=",",skiprows=1).join(pd.read_csv('puma-s1-2-repeat-1.csv',
                    skiprows=1).iloc[:,3:],rsuffix='2').join(pd.read_csv('puma-s1-3-repeat-1.csv',
                    skiprows=1).iloc[:,3:],rsuffix='3')
PUMAS1_data_2=pd.read_csv("puma-s1-1-repeat-2.csv",sep=",",skiprows=1).join(pd.read_csv('puma-s1-2-repeat-2.csv',
                    skiprows=1).iloc[:,3:],rsuffix='2').join(pd.read_csv('puma-s1-3-repeat-2.csv',
                    skiprows=1).iloc[:,3:],rsuffix='3')
PUMAS2_data_1=pd.read_csv("puma-s2-1-repeat-1.csv",sep=",",skiprows=1).join(pd.read_csv('puma-s2-2-repeat-1.csv',
                    skiprows=1).iloc[:,3:],rsuffix='2').join(pd.read_csv('puma-s2-3-repeat-1.csv',
                    skiprows=1).iloc[:,3:],rsuffix='3')
PUMAS2_data_2=pd.read_csv("puma-s2-1-repeat-2.csv",sep=",",skiprows=1).join(pd.read_csv('puma-s2-2-repeat-2.csv',
                    skiprows=1).iloc[:,3:],rsuffix='2').join(pd.read_csv('puma-s2-3-repeat-2.csv',
                    skiprows=1).iloc[:,3:],rsuffix='3')
PUMAS3_data_1=pd.read_csv("puma-s3-1-repeat-1.csv",sep=",",skiprows=1).join(pd.read_csv('puma-s3-2-repeat-1.csv',
                    skiprows=1).iloc[:,3:],rsuffix='2').join(pd.read_csv('puma-s3-3-repeat-1.csv',
                    skiprows=1).iloc[:,3:],rsuffix='3')
PUMAS3_data_2=pd.read_csv("puma-s3-1-repeat-2.csv",sep=",",skiprows=1).join(pd.read_csv('puma-s3-2-repeat-2.csv',
                    skiprows=1).iloc[:,3:],rsuffix='2').join(pd.read_csv('puma-s3-3-repeat-2.csv',
                    skiprows=1).iloc[:,3:],rsuffix='3')
Ash1_data_1=pd.read_csv("ash1-1-repeat-1.csv",sep=",",skiprows=1).join(pd.read_csv('ash1-2-repeat-1.csv',
                    skiprows=1).iloc[:,3:],rsuffix='2').join(pd.read_csv('ash1-3-repeat-1.csv',
                    skiprows=1).iloc[:,3:],rsuffix='3')
Ash1_data_2=pd.read_csv("ash1-1-repeat-2.csv",sep=",",skiprows=1).join(pd.read_csv('ash1-2-repeat-2.csv',
                    skiprows=1).iloc[:,3:],rsuffix='2').join(pd.read_csv('ash1-3-repeat-2.csv',
                    skiprows=1).iloc[:,3:],rsuffix='3')
E1A_data_1=pd.read_csv("e1a-1-repeat-1.csv",sep=",",skiprows=1).join(pd.read_csv('e1a-2-repeat-1.csv',
                    skiprows=1).iloc[:,3:],rsuffix='2').join(pd.read_csv('e1a-3-repeat-1.csv',
                    skiprows=1).iloc[:,3:],rsuffix='3')
E1A_data_2=pd.read_csv("e1a-1-repeat-2.csv",sep=",",skiprows=1).join(pd.read_csv('e1a-2-repeat-2.csv',
                    skiprows=1).iloc[:,3:],rsuffix='2').join(pd.read_csv('e1a-3-repeat-2.csv',
                    skiprows=1).iloc[:,3:],rsuffix='3')
P53_data_1=pd.read_csv("p53-1-repeat-1.csv",sep=",",skiprows=1).join(pd.read_csv('p53-2-repeat-1.csv',
                    skiprows=1).iloc[:,3:],rsuffix='2').join(pd.read_csv('p53-3-repeat-1.csv',
                    skiprows=1).iloc[:,3:],rsuffix='3')
P53_data_2=pd.read_csv("p53-1-repeat-2.csv",sep=",",skiprows=1).join(pd.read_csv('p53-2-repeat-2.csv',
                    skiprows=1).iloc[:,3:],rsuffix='2').join(pd.read_csv('p53-3-repeat-2.csv',
                    skiprows=1).iloc[:,3:],rsuffix='3')
FUS_data_1=pd.read_csv("fus-1-repeat-1.csv",sep=",",skiprows=1)
                                                                                             
# Build a data frame with correction factors for the donor and acceptor at each concentration of each solute
base_corr_fact_df=build_base_correction_factor_df(TQ_data, NG_data)

# Get the average IDR/dirA intensity ratio for normalization of experimental spectra
avg_idr_dirA_intensity_ratio = get_avg_idr_dirA_intensity_ratio()
                    
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
PUMA_efret_1=get_efret('PUMA WT',PUMA_data_1,range(3,PUMA_data_1.shape[1]),base_corr_fact_df,TQ_data,NG_data,True,avg_idr_dirA_intensity_ratio)
PUMA_efret_2=get_efret('PUMA WT',PUMA_data_2,range(3,PUMA_data_2.shape[1]),base_corr_fact_df,TQ_data,NG_data,True,avg_idr_dirA_intensity_ratio)
PUMAS1_efret_1=get_efret('PUMA S1',PUMAS1_data_1,range(3,PUMAS1_data_1.shape[1]),base_corr_fact_df,TQ_data,NG_data,True,avg_idr_dirA_intensity_ratio)
PUMAS1_efret_2=get_efret('PUMA S1',PUMAS1_data_2,range(3,PUMAS1_data_2.shape[1]),base_corr_fact_df,TQ_data,NG_data,True,avg_idr_dirA_intensity_ratio)
PUMAS2_efret_1=get_efret('PUMA S2',PUMAS2_data_1,range(3,PUMAS2_data_1.shape[1]),base_corr_fact_df,TQ_data,NG_data,True,avg_idr_dirA_intensity_ratio)
PUMAS2_efret_2=get_efret('PUMA S2',PUMAS2_data_2,range(3,PUMAS2_data_2.shape[1]),base_corr_fact_df,TQ_data,NG_data,True,avg_idr_dirA_intensity_ratio)
PUMAS3_efret_1=get_efret('PUMA S3',PUMAS3_data_1,range(3,PUMAS3_data_1.shape[1]),base_corr_fact_df,TQ_data,NG_data,True,avg_idr_dirA_intensity_ratio)
PUMAS3_efret_2=get_efret('PUMA S3',PUMAS3_data_2,range(3,PUMAS3_data_2.shape[1]),base_corr_fact_df,TQ_data,NG_data,True,avg_idr_dirA_intensity_ratio)
Ash1_efret_1=get_efret('Ash1',Ash1_data_1,range(3,Ash1_data_1.shape[1]),base_corr_fact_df,TQ_data,NG_data,True,avg_idr_dirA_intensity_ratio)
Ash1_efret_2=get_efret('Ash1',Ash1_data_2,range(3,Ash1_data_2.shape[1]),base_corr_fact_df,TQ_data,NG_data,True,avg_idr_dirA_intensity_ratio)
E1A_efret_1=get_efret('E1A',E1A_data_1,range(3,E1A_data_1.shape[1]),base_corr_fact_df,TQ_data,NG_data,True,avg_idr_dirA_intensity_ratio)
E1A_efret_2=get_efret('E1A',E1A_data_2,range(3,E1A_data_2.shape[1]),base_corr_fact_df,TQ_data,NG_data,True,avg_idr_dirA_intensity_ratio)
P53_efret_1=get_efret('p53',P53_data_1,range(3,P53_data_1.shape[1]),base_corr_fact_df,TQ_data,NG_data,True,avg_idr_dirA_intensity_ratio)
P53_efret_2=get_efret('p53',P53_data_2,range(3,P53_data_2.shape[1]),base_corr_fact_df,TQ_data,NG_data,True,avg_idr_dirA_intensity_ratio)
FUS_efret_1=get_efret('FUS',FUS_data_1,range(3,FUS_data_1.shape[1]),base_corr_fact_df,TQ_data,NG_data,True,avg_idr_dirA_intensity_ratio)

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
puma_wt_concat = pd.concat([PUMA_efret_1[PUMA_efret_1.sol=='Buffer']['fit'].item(),
                        PUMA_efret_2[PUMA_efret_2.sol=='Buffer']['fit'].item()])
puma_s1_concat = pd.concat([PUMAS1_efret_1[PUMAS1_efret_1.sol=='Buffer']['fit'].item(),
                        PUMAS1_efret_2[PUMAS1_efret_2.sol=='Buffer']['fit'].item()])
puma_s2_concat = pd.concat([PUMAS2_efret_1[PUMAS2_efret_1.sol=='Buffer']['fit'].item(),
                        PUMAS2_efret_2[PUMAS2_efret_2.sol=='Buffer']['fit'].item()])
puma_s3_concat = pd.concat([PUMAS3_efret_1[PUMAS3_efret_1.sol=='Buffer']['fit'].item(),
                        PUMAS3_efret_2[PUMAS3_efret_2.sol=='Buffer']['fit'].item()])
ash1_concat = pd.concat([Ash1_efret_1[Ash1_efret_1.sol=='Buffer']['fit'].item(),
                        Ash1_efret_2[Ash1_efret_2.sol=='Buffer']['fit'].item()])
e1a_concat = pd.concat([E1A_efret_1[E1A_efret_1.sol=='Buffer']['fit'].item(),
                        E1A_efret_2[E1A_efret_2.sol=='Buffer']['fit'].item()])
p53_concat = pd.concat([P53_efret_1[P53_efret_1.sol=='Buffer']['fit'].item(),
                        P53_efret_2[P53_efret_2.sol=='Buffer']['fit'].item()])

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
                    "fit_GS48":gs48_concat.groupby(gs48_concat.index).mean(),
                    "fit_puma_wt":puma_wt_concat.groupby(puma_wt_concat.index).mean(),
                    "fit_puma_s1":puma_s1_concat.groupby(puma_s1_concat.index).mean(),
                    "fit_puma_s2":puma_s2_concat.groupby(puma_s2_concat.index).mean(),
                    "fit_puma_s3":puma_s3_concat.groupby(puma_s3_concat.index).mean(),
                    "fit_ash1":ash1_concat.groupby(ash1_concat.index).mean(),
                    "fit_e1a":e1a_concat.groupby(e1a_concat.index).mean(),
                    "fit_p53":p53_concat.groupby(p53_concat.index).mean(),
                    "fit_fus":FUS_efret_1[FUS_efret_1.sol=='Buffer']['fit'].item() })

# Do a linear regression of the intensity values of the GS linkers at each wavelength
# and calculate a slope and y-intercept for each wavelength for use in calculating 
# GS-equivalent spectra imaginary GS linkers of different lengths (N)
gs_equiv_regression_slopes = np.array([])
gs_equiv_regression_intercepts = np.array([])
linker_Ns = [0, 16, 32, 48, 64, 96]
for i in range(len(df.fit_GS0)):
    # These indices start at 1 instead of 0.
    intensities = [df.fit_GS0[i+1], df.fit_GS8[i+1], df.fit_GS16[i+1], 
                   df.fit_GS24[i+1], df.fit_GS32[i+1], df.fit_GS48[i+1]]
    # Do linear regression for N vs. fluorescence intensity.
    a_fit,cov=curve_fit(linearFunc,linker_Ns,intensities)
    # a_fit[1] = slope, a_fit[0] = intercept (for the regression at this wavelength)
    gs_equiv_regression_slopes = np.append(gs_equiv_regression_slopes, a_fit[1])
    gs_equiv_regression_intercepts = np.append(gs_equiv_regression_intercepts, a_fit[0])

def get_gs_equiv_fluor_spect(N, regression_slopes, regression_intercepts):
    gs_equiv_spectrum = np.array([])
    for i in range(len(df.fit_GS0)):
        gs_equiv_spectrum = np.append(gs_equiv_spectrum, regression_intercepts[i] + (N * regression_slopes[i]))
    return (gs_equiv_spectrum)

df_cols = ['fit_GS0', 'fit_GS8', 'fit_GS16', 'fit_GS24', 'fit_GS32', 'fit_GS48', 'fit_puma_wt', 'fit_puma_s1', 'fit_puma_s2', 'fit_puma_s3', 'fit_ash1', 'fit_e1a', 'fit_p53', 'fit_fus']
labels = ['GS0','GS8','GS16','GS24','GS32','GS48','WT PUMA', 'PUMA S1', 'PUMA S2', 'PUMA S3', 'Ash1', 'E1A', 'p53', 'FUS']  
colors = ['black','dimgrey','grey','darkgrey','silver','lightgray','darkslateblue','blueviolet','violet','darkmagenta','limegreen','teal','royalblue','dodgerblue']
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
        if index < len(df_cols):
            N = Ns[index]
            axs[i][j].plot(df.wavelengths, 100 * (df[df_cols[index]] / df[df_cols[index]][26]), 
                            c=colors[index], label=labels[index], linewidth=3, linestyle='-', zorder=100)
            if labels[index] not in ['GS0','GS8','GS16','GS24','GS32','GS48']:
                gs_equiv_spectrum = get_gs_equiv_fluor_spect(N, gs_equiv_regression_slopes, gs_equiv_regression_intercepts)
                axs[i][j].plot(df.wavelengths, 100 * gs_equiv_spectrum / gs_equiv_spectrum[26], 
                               c='black', linewidth=2, linestyle='dotted', zorder=50)
            axs[i][j].plot(df.wavelengths, 5000 * df.f_d / df.f_d.sum(), c='cyan')
            axs[i][j].plot(df.wavelengths, 3000 * df.f_a / df.f_a.sum(), c='lime')
            axs[i][j].fill_between(df.wavelengths, 5000 * df.f_d / df.f_d.sum(), color='cyan', alpha=0.5)
            axs[i][j].fill_between(df.wavelengths, 3000 * df.f_a / df.f_a.sum(), color='lime', alpha=0.3)
            axs[i][j].legend(fontsize=30, loc='upper right')
            axs[i][j].xaxis.tick_bottom()
            axs[i][j].yaxis.tick_left()
            axs[i][j].set_xlabel('wavelength (nm)', fontsize=36, labelpad=0)
            axs[i][j].set_ylabel('fluorescence intensity (AU)', fontsize=36, labelpad=0)
            axs[i][j].set_xlim([450, 600])
        else:
            axs[i][j].axis('off')
plt.savefig('hidden_structure_fig_s3.png', bbox_inches='tight')
plt.show()
