########################################################################################################
# hidden_structure_fig_2b_2c_2d_2e.py
# 
# Author: David Moses (dmoses5@ucmerced.edu)
# 
# Supporting material for "Hidden structure in disordered proteins is adaptive to intracellular changes"
# 
# https://www.biorxiv.org/content/10.1101/2021.11.24.469609v1
# 
# Figure 2: R_e, R_g and solution space scans of naturally occurring IDRs
#
# 2A: Cartoons of different constructs in the same buffer solution.
#
# 2B: R_g vs. N. 2C: R_e vs. N. 2D: R_e, R_g, GS-equivalents. 2E: Distance from GS-equivalents.
#
# 2E: Cartoons of the same construct in two different solution conditions.
#
# 2F: Solution-space scans. (Produced by a different script.)
#
# Required file: hidden_structure_fret_and_saxs_aggregated_data.csv
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

# for use with curve_fit
def linearFunc(x,intercept,slope):
    y = intercept + slope * x
    return y

# Create and format the figure
ncols=2
nrows=2
wspace=0
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=False, sharey=False, 
                        figsize=(ncols*10.5 + (wspace*(ncols-1)), nrows*10))
plt.subplots_adjust(wspace=wspace)
plt.margins(x=0,y=0)

data_df = pd.read_csv('hidden_structure_fret_and_saxs_aggregated_data.csv')

###############
# Fig 2A, 2B
###############
# N vs R_g or R_e for idrs
idrs = ['PUMA WT','p53', 'FUS', 'E1A', 'Ash1']
idr_titles = ['PUMA', 'p53', 'FUS', 'E1A', 'Ash1']
idr_Ns = [34, 61, 163, 40, 83]
idr_colors = ['darkslateblue','royalblue','dodgerblue','teal','limegreen']
for i in range(len(idrs)):
    R_g = data_df[(data_df.idr == idrs[i]) & (data_df.sol=='Buffer (standard)')].R_g_mean
    R_g_err = data_df[(data_df.idr == idrs[i]) & (data_df.sol=='Buffer (standard)')].R_g_err
    axs[0][0].scatter(idr_Ns[i],R_g,s=1800,marker='o',c=idr_colors[i],edgecolors=None,linewidth=None,label=idr_titles[i],alpha=1,zorder=75)
    axs[0][0].errorbar(idr_Ns[i],R_g,yerr=R_g_err,capsize=7,capthick=2,ls='none',color='darkgray',elinewidth=3, zorder=100)
    R_e = data_df[(data_df.idr == idrs[i]) & (data_df.sol=='Buffer (standard)')].R_e_mean
    R_e_err = data_df[(data_df.idr == idrs[i]) & (data_df.sol=='Buffer (standard)')].R_e_err
    axs[0][1].scatter(idr_Ns[i],R_e,s=1800,marker='o',c=idr_colors[i],edgecolors=None,linewidth=None,label=idr_titles[i],alpha=1,zorder=75)
    axs[0][1].errorbar(idr_Ns[i],R_e,yerr=R_e_err,capsize=7,capthick=2,ls='none',color='darkgray',elinewidth=3, zorder=100)

# N vs R_g or R_e regression lines for GS linkers
linkers = ['GS0','GS8','GS16','GS24','GS32','GS48']
linker_Ns = [0, 16, 32, 48, 64, 96]
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
# linear regression for N vs. R_e
a_fit,cov=curve_fit(linearFunc,linker_Ns,linker_R_es,sigma=linker_R_e_errs)
R_e_intercept = a_fit[0]
R_e_slope = a_fit[1]
# linear regression for N vs. R_g
a_fit,cov=curve_fit(linearFunc,linker_Ns,linker_R_gs,sigma=linker_R_g_errs)
R_g_intercept = a_fit[0]
R_g_slope = a_fit[1]
# N vs R_g line
axs[0][0].set_xlim(25,172)
linkers_R_g_x_vals = np.array(axs[0][0].get_xlim())
linkers_R_g_y_vals = R_g_intercept + (R_g_slope * linkers_R_g_x_vals)
axs[0][0].plot(linkers_R_g_x_vals, linkers_R_g_y_vals, c='cornflowerblue', linestyle='dashed', linewidth=4, zorder=50, label='GS(N)')
# N vs R_e line
axs[0][1].set_xlim(25,172)
linkers_R_e_x_vals = np.array(axs[0][1].get_xlim())
linkers_R_e_y_vals = R_e_intercept + (R_e_slope * linkers_R_e_x_vals)
axs[0][1].plot(linkers_R_e_x_vals, linkers_R_e_y_vals, c='cornflowerblue', linestyle='dashed', linewidth=4, zorder=50, label='GS(N)')

for i in range(2):
    axs[0][i].set_xlabel('N$_{residues}$')
axs[0][0].legend(loc='upper left', prop={'size': 30}, handletextpad=0.1, frameon=False)
axs[0][0].set_ylabel('$R_g$ $(\AA)$')
axs[0][1].set_ylabel('$R^{app}_e$ $(\AA)$')

###############
# Fig 2C
###############

idrs = ['PUMA WT','p53', 'FUS', 'E1A', 'Ash1']
idr_titles = ['PUMA', 'p53', 'FUS', 'E1A', 'Ash1']
idr_Ns = [34, 61, 163, 40, 83]
idr_colors = ['darkslateblue','royalblue','dodgerblue','teal','limegreen']
linkers = ['GS0','GS8','GS16','GS24','GS32','GS48']
linker_Ns = [0, 16, 32, 48, 64, 96]
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
# axs[1][0].set_ylim(49,80.5)
axs[1][0].set_xlim(30,53)
line_x_vals = np.array(axs[1][0].get_xlim())
line_y_vals = myoutput.beta[1] + (myoutput.beta[0] * line_x_vals)
axs[1][0].plot(line_x_vals, line_y_vals, c='cornflowerblue', linestyle='dashed',linewidth=4, zorder=50,label='GS')
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
for i in range(len(idrs)):
    R_e = data_df[(data_df.idr == idrs[i]) & (data_df.sol=='Buffer (standard)')].R_e_mean
    R_e_err = data_df[(data_df.idr == idrs[i]) & (data_df.sol=='Buffer (standard)')].R_e_err
    R_g = data_df[(data_df.idr == idrs[i]) & (data_df.sol=='Buffer (standard)')].R_g_mean
    R_g_err = data_df[(data_df.idr == idrs[i]) & (data_df.sol=='Buffer (standard)')].R_g_err
    # actual
    axs[1][0].scatter(R_g,R_e,s=2000,marker='o',c=idr_colors[i],edgecolors=None,linewidth=None,label='IDR',alpha=1,zorder=50)
    axs[1][0].errorbar(R_g,R_e,xerr=R_g_err,yerr=R_e_err,capsize=7,capthick=2,ls='none',color='darkgray',elinewidth=3,zorder=100)
    # interpolated gs-equivalent
    interpolated_R_g = R_g_intercept + (R_g_slope * idr_Ns[i])
    interpolated_R_g_err = sqrt((idr_Ns[i]**2 * R_g_slope_err**2) + R_g_intercept_err**2)
    interpolated_R_e = R_e_intercept+(R_e_slope*idr_Ns[i])
    interpolated_R_e_err = sqrt((idr_Ns[i]**2 * R_e_slope_err**2) + R_e_intercept_err**2)
    axs[1][0].scatter(interpolated_R_g, interpolated_R_e, s=1600,marker='o',c='white',edgecolors=idr_colors[i],linewidth=5,alpha=1,label='GS-equiv',zorder=75)
    axs[1][0].errorbar(interpolated_R_g, interpolated_R_e, xerr=interpolated_R_g_err, yerr=interpolated_R_e_err,capsize=7,capthick=2,ls='none',color='darkgray',elinewidth=3,zorder=100)
axs[1][0].set_xlabel('$R_g$ $(\AA)$', labelpad=-10)
axs[1][0].set_ylabel('$R^{app}_e$ $(\AA)$', rotation=90)
axs[1][0].label_outer()
handles, labels = axs[1][0].get_legend_handles_labels()
axs[1][0].legend(handles[1:3], labels[1:3], loc='lower right', fontsize=40, handletextpad=0.1, frameon=False)
axs[1][0].legend_.legendHandles[0].set_color('black')
axs[1][0].legend_.legendHandles[1].set_edgecolor('black')

###############
# Fig 2D
###############

idrs = ['PUMA WT','p53', 'FUS', 'E1A', 'Ash1']
idr_Ns = [34, 61, 163, 40, 83] 
idr_titles = ['PUMA', 'p53', 'FUS', 'E1A', 'Ash1']
idr_colors = ['darkslateblue','royalblue','dodgerblue','teal','limegreen']
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
                                           alpha = 0.8, error_kw=dict(ecolor='k'),ax=axs[1][1], rot=0, 
                                           color={'R_g_dists':idr_colors, 'R_e_dists':idr_colors},
                                           width = 0.8, edgecolor = 'black', linewidth = 1)
bars = axs[1][1].patches
for i in range(int(len(bars)/2)):
    bars[i].set_hatch('') 
for i in range(int(len(bars)/2),len(bars)):
    bars[i].set_hatch('///') 
axs[1][1].set_xlabel('IDR', labelpad=10)
axs[1][1].set_ylabel('$\Delta$$R$ $(\AA)$', rotation=90, labelpad=-20)
for tick in axs[1][1].xaxis.get_major_ticks():
    tick.label.set_fontsize(36)
axs[1][1].legend(['$R_g$', '$R_e^{app}$'], fontsize=40, loc='lower left', handletextpad=0.5, frameon=False)
axs[1][1].axhline(0, color='black', lw=2, alpha=1, linestyle='dashed')
x_tick_vals = axs[1][1].get_xticks()
# for i in range(len(idr_Ns)):
#     axs[1][1].annotate('$N_{res}$=' + str(idr_Ns[i]), (x_tick_vals[i], 1), fontsize=22, ha='center')
for i in [0,2,3,4]:
    axs[1][1].annotate('$N_{res}$=' + str(idr_Ns[i]), (x_tick_vals[i], 0.2), fontsize=22, ha='center')
axs[1][1].annotate('$N_{res}$=' + str(idr_Ns[1]), (x_tick_vals[1], -2.5), fontsize=22, ha='center')

fig.tight_layout()
plt.savefig('hidden_structure_fig_2b_2c_2d_2e.png', bbox_inches='tight')
plt.show()

