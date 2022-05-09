########################################################################################################
# hidden_structure_fig_3c_3d_3e.py
# 
# Author: David Moses (dmoses5@ucmerced.edu)
# 
# Supporting material for "Hidden structure in disordered proteins is adaptive to intracellular changes"
# 
# https://www.biorxiv.org/content/10.1101/2021.11.24.469609v1
# 
# Figure 3: Molecular features, CD, R_e, R_g and solution space scans of WT PUMA BH3 and scrambles
#
# 3A: Sequences of WT PUMA and sequence shuffles. 3B: Molecular features of WT PUMA and shuffles.
#
# 3C: CD spectroscopy. 3D: R_e, R_g, GS-equivalent. 3E: Distance from GS-equivalents. (This script.)
#
# 3F: Solution-space scans.
########################################################################################################

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from math import sqrt
import numpy as np
from scipy.optimize import curve_fit
import scipy.odr.odrpack as odrpack

mpl.style.use('default')
mpl.rcParams['axes.linewidth'] = 7 #set the value globally
mpl.rcParams['xtick.major.size'] = 20
mpl.rcParams['xtick.major.width'] = 7
mpl.rcParams['xtick.minor.size'] = 10
mpl.rcParams['xtick.minor.width'] = 7
mpl.rcParams['ytick.major.size'] = 20
mpl.rcParams['ytick.major.width'] = 7
mpl.rcParams['ytick.labelsize'] = 55
mpl.rcParams['xtick.labelsize'] = 55
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

# for use with curve_fit
def linearFunc(x,intercept,slope):
    y = intercept + slope * x
    return y

# Create and format the figure
ncols=3
nrows=1
wspace=0
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=False, sharey=False, 
                        figsize=(ncols*10 + (wspace*(ncols-1)), nrows*10.4))
plt.subplots_adjust(wspace=wspace)
plt.margins(x=0,y=0)

data_df = pd.read_csv('hidden_structure_fret_and_saxs_aggregated_data.csv')

###############
# Fig 3B
###############

csvs = ['PUMAWT_20_20um-1MRE.csv', 'S1_20_20um-1MRE.csv', 'S2_20_20um-1MRE.csv', 'S3_20_20um-1MRE.csv']
labels = ['WT', 'S1', 'S2', 'S3']
colors = ['darkslateblue','blueviolet','violet','darkmagenta']
for i in range(len(csvs)):
    df = pd.read_csv(csvs[i], names=['wavelength', 'theta'])
    wavelengths = df.wavelength[:651]
    thetas = df.theta[:651]
    axs[0].plot(wavelengths, thetas, color=colors[i], label=labels[i], linestyle='-', linewidth=7, alpha=0.9)
axs[0].set_xlabel('wavelength (nm)')
axs[0].set_ylabel('MRE (10$^3$ deg cm$^2$ dmol$^{-1}$)', rotation=90, labelpad=-10, fontsize=40)
axs[0].set_xlim([195, 260])
axs[0].legend(fontsize=36, frameon=False) 

###############
# Fig 3C
###############

idrs = ['PUMA WT','PUMA S1','PUMA S2','PUMA S3']
idr_titles = ['WT','S1','S2','S3']
gs_equiv_title = 'GS'
idr_N = 34
idr_colors = ['darkslateblue','blueviolet','violet','darkmagenta']
linkers = ['GS0','GS8','GS16','GS24','GS32','GS48']
linker_Ns = [0,16,32,48,64,96]
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
# use scipy.odr to draw the linkers regression line and confidence interval
linear = odrpack.Model(f)
# mydata = odrpack.RealData(x, y, sx=sx, sy=sy)
mydata = odrpack.RealData(linker_R_gs, linker_R_es, sx=linker_R_g_errs, sy=linker_R_e_errs)
# instantiate ODR with your data, model and initial parameter estimate
myodr = odrpack.ODR(mydata, linear, beta0=[1., 2.])
myoutput = myodr.run()
axs[1].set_xlim([31, 37])
line_x_vals = np.array(axs[1].get_xlim())
line_y_vals = myoutput.beta[1] + (myoutput.beta[0] * line_x_vals)
axs[1].plot(line_x_vals, line_y_vals, c='cornflowerblue', linestyle='dashed', linewidth=4, zorder=50)
# linear regression for N vs. R_e
# error for interpolated R_e or R_g will be sqrt(N**2 * slope_err**2 + intercept_err**2)
# one reference for this is www.itl.nist.gov/div898/handbook/mpc/section5/mpc552.htm
a_fit,cov=curve_fit(linearFunc,linker_Ns,linker_R_es)
R_e_intercept = a_fit[0]
R_e_slope = a_fit[1]
R_e_intercept_err = sqrt(cov[0][0])
R_e_slope_err = sqrt(cov[1][1])
# linear regression for N vs. R_g
a_fit,cov=curve_fit(linearFunc,linker_Ns,linker_R_gs)
R_g_intercept = a_fit[0]
R_g_slope = a_fit[1]
R_g_intercept_err = sqrt(cov[0][0])
R_g_slope_err = sqrt(cov[1][1])

# plot actual Re and Rg
for i in range(len(idrs)):
    R_e = data_df[(data_df.idr == idrs[i]) & (data_df.sol=='Buffer (standard)')].R_e_mean
    R_e_err = data_df[(data_df.idr == idrs[i]) & (data_df.sol=='Buffer (standard)')].R_e_err
    R_g = data_df[(data_df.idr == idrs[i]) & (data_df.sol=='Buffer (standard)')].R_g_mean
    R_g_err = data_df[(data_df.idr == idrs[i]) & (data_df.sol=='Buffer (standard)')].R_g_err
    axs[1].scatter(R_g,R_e,s=2000,marker='o',c=idr_colors[i],edgecolors=None,linewidth=None,label=idr_titles[i],alpha=1,zorder=50)
    axs[1].errorbar(R_g,R_e,xerr=R_g_err,yerr=R_e_err,capsize=7,capthick=2,ls='none',color='darkgray',elinewidth=3, zorder=100)

# plot interpolated gs-equivalent
interpolated_R_g = R_g_intercept + (R_g_slope * idr_N)
interpolated_R_g_err = sqrt((idr_N**2 * R_g_slope_err**2) + R_g_intercept_err**2)
interpolated_R_e = R_e_intercept+(R_e_slope*idr_N)
interpolated_R_e_err = sqrt((idr_N**2 * R_e_slope_err**2) + R_e_intercept_err**2)
axs[1].scatter(interpolated_R_g, interpolated_R_e, s=1600,marker='o',c='white',edgecolors='black',linewidth=5,alpha=1,label=gs_equiv_title)
axs[1].errorbar(interpolated_R_g, interpolated_R_e, xerr=interpolated_R_g_err, yerr=interpolated_R_e_err, capsize=7,capthick=2,ls='none',color='darkgray',elinewidth=3, zorder=100)

axs[1].set_xlabel('$R_g$ $(\AA)$', labelpad=0)
axs[1].set_ylabel('$R^{app}_e$ $(\AA)$', rotation=90)
axs[1].legend(loc='upper left', prop={'size': 32}, handletextpad=0.1, frameon=False)

###############
# Fig 3D
###############

idrs = ['PUMA WT','PUMA S1','PUMA S2','PUMA S3']
idr_Ns = [34, 34, 34, 34]
idr_titles = ['WT','S1','S2','S3']
idr_colors = ['darkslateblue','blueviolet','violet','darkmagenta']
linkers = ['GS0','GS8','GS16','GS24','GS32','GS48']
linker_Ns = [0,16,32,48,64,96]
linker_R_gs = np.array([])
linker_R_g_errs = np.array([])
linker_R_es = np.array([])
linker_R_e_errs = np.array([])
for i in range(len(linkers)):
    R_g = data_df[(data_df.idr == linkers[i]) & (data_df.sol=='Buffer (standard)')].R_g_mean
    R_g_err = data_df[(data_df.idr == linkers[i]) & (data_df.sol=='Buffer (standard)')].R_g_err
    R_e = data_df[(data_df.idr == linkers[i]) & (data_df.sol=='Buffer (standard)')].R_e_mean
    R_e_err = data_df[(data_df.idr == linkers[i]) & (data_df.sol=='Buffer (standard)')].R_e_err
    linker_R_gs = np.append(linker_R_gs,R_g)
    linker_R_g_errs = np.append(linker_R_g_errs,R_g_err)
    linker_R_es = np.append(linker_R_es,R_e)
    linker_R_e_errs = np.append(linker_R_e_errs,R_e_err)
# linear regression for N vs. R_e
# error for interpolated R_e or R_g will be sqrt(N**2 * slope_err**2 + intercept_err**2)
# one reference for this is www.itl.nist.gov/div898/handbook/mpc/section5/mpc552.htm
a_fit,cov=curve_fit(linearFunc,linker_Ns,linker_R_es,sigma=linker_R_e_errs)
linker_R_e_intercept = a_fit[0]
linker_R_e_slope = a_fit[1]
linker_R_e_intercept_err = sqrt(cov[0][0])
linker_R_e_slope_err = sqrt(cov[1][1])
# linear regression for N vs. R_g
a_fit,cov=curve_fit(linearFunc,linker_Ns,linker_R_gs,sigma=linker_R_g_errs)
linker_R_g_intercept = a_fit[0]
linker_R_g_slope = a_fit[1]
linker_R_g_intercept_err = sqrt(cov[0][0])
linker_R_g_slope_err = sqrt(cov[1][1])
R_g_dists = []
R_g_dist_errs = []
R_e_dists = []
R_e_dist_errs = []
# interpolate gs-equivalents and calculate distance between them and actual points
for i in range(len(idrs)):
    actual_R_g = data_df[(data_df.idr == idrs[i]) & (data_df.sol=='Buffer (standard)')].R_g_mean.values[0]
    actual_R_g_err = data_df[(data_df.idr == idrs[i]) & (data_df.sol=='Buffer (standard)')].R_g_err
    actual_R_e = data_df[(data_df.idr == idrs[i]) & (data_df.sol=='Buffer (standard)')].R_e_mean.values[0]
    actual_R_e_err = data_df[(data_df.idr == idrs[i]) & (data_df.sol=='Buffer (standard)')].R_e_err
    gs_equiv_R_g = linker_R_g_intercept + (linker_R_g_slope * idr_Ns[i])
    gs_equiv_R_g_err = sqrt((idr_Ns[i]**2 * linker_R_g_slope_err**2) + linker_R_g_intercept_err**2)
    gs_equiv_R_e = linker_R_e_intercept + (linker_R_e_slope * idr_Ns[i])
    gs_equiv_R_e_err = sqrt((idr_Ns[i]**2 * linker_R_e_slope_err**2) + linker_R_e_intercept_err**2)
    R_g_dists.append(actual_R_g - gs_equiv_R_g)
    R_e_dists.append(actual_R_e - gs_equiv_R_e)
    R_g_dist_errs.append(sqrt(actual_R_g_err**2 + gs_equiv_R_g_err**2))
    R_e_dist_errs.append(sqrt(actual_R_e_err**2 + gs_equiv_R_e_err**2))
# the new way
df = pd.DataFrame(R_g_dists, columns=['R_g_dists'], index=idr_titles)
df['R_e_dists'] = R_e_dists 
df['R_g_dist_errs'] = R_g_dist_errs 
df['R_e_dist_errs'] = R_e_dist_errs
df[['R_g_dists', 'R_e_dists']].plot(kind='bar', yerr=df[['R_g_dist_errs', 'R_e_dist_errs']].values.T, 
                                           alpha = 0.8, error_kw=dict(ecolor='k'),ax=axs[2], rot=0, 
                                           color={'R_g_dists':idr_colors, 'R_e_dists':idr_colors},
                                           width = 0.8, edgecolor = 'black', linewidth = 1)
bars = axs[2].patches
for i in range(int(len(bars)/2)):
    bars[i].set_hatch('') 
for i in range(int(len(bars)/2),len(bars)):
    bars[i].set_hatch('///') 
axs[2].set_xlabel('IDR', labelpad=10)
axs[2].set_ylabel('$\Delta$$R$ $(\AA)$', rotation=90, labelpad=0)
axs[2].legend(['$R_g$', '$R_e^{app}$'], fontsize=40, loc='lower right', handletextpad=0.5, frameon=False)
axs[2].axhline(0, color='black', lw=2, alpha=1, linestyle='dashed')
x_tick_vals = axs[2].get_xticks()
axs[2].annotate('all IDRs', (1.2, -3.9), fontsize=32, ha='center')
axs[2].annotate('$N_{res}$=34', (1.2, -4.5), fontsize=32, ha='center')

fig.tight_layout()
plt.savefig('hidden_structure_fig_3c_3d_3e.png', bbox_inches='tight')
plt.show()

