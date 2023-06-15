########################################################################################################
# hidden_structure_fig_s5.py
# 
# Author: David Moses (dmoses5@ucmerced.edu)
# 
# Supporting material for "Hidden structure in disordered proteins is adaptive to intracellular changes"
# 
# https://www.biorxiv.org/content/10.1101/2021.11.24.469609v1
# 
# Figures 2G, 3F, S5: Heatmap of Re values for various IDRs in various solution conditions
#
# 2G: Naturally occurring IDRs. 3F: PUMA BH3 and scrambles. S5: GS polymers of various lengths.
#
# Choice of IDR_SET determines which figure is generated.
#
# Required file: hidden_structure_fret_and_saxs_aggregated_data.csv
########################################################################################################

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit

mpl.style.use('default')
plt.rc('xtick', labelsize=45) 
plt.rc('ytick', labelsize=45) 
plt.rcParams["xtick.major.size"] = 10
plt.rcParams["ytick.major.size"] = 10
mpl.rcParams['xtick.major.width'] = 4
mpl.rcParams['ytick.major.width'] = 4
mpl.rcParams['xtick.direction'] = 'out'
mpl.rcParams['ytick.direction'] = 'out'
mpl.rcParams['axes.linewidth'] = 3
mpl.rc('axes',edgecolor='white')
sns.set_style("white")

########################################################################################################################
# Choose whether to make figure for naturally occurring IDRs or PUMAs -- keep one, comment out the others
########################################################################################################################
# IDR_SET = 'idrs' # Uncomment for Fig 2G
# IDR_SET = 'pumas' # Uncomment for Fig 3F
IDR_SET = 'gs' # Uncomment for Fig S5

###########################
# Make colormap for heatmap
###########################
if IDR_SET == 'idrs':
    minl=-50
    maxl=50
elif IDR_SET == 'pumas':
    minl=-45
    maxl=45
elif IDR_SET == 'gs':
    minl=-40
    maxl=40

idr_to_N = {
    "GS0": 0, 
    "GS8": 16, 
    "GS16": 32, 
    "GS24": 48, 
    "GS32": 64, 
    "GS48": 96, 
    "PUMA WT": 34,
    "PUMA S1": 34,
    "PUMA S2": 34,
    "PUMA S3": 34,
    "E1A": 40,
    "Ash1": 83,
    "FUS": 163,
    "p53": 61
}

solute_to_axis_label = {
    "Ficoll": '(monoM)', 
    "PEG2000": '(monoM)', 
    "Glycine": '(M)', 
    "Sarcosine": '(M)', 
    "Urea": '(M)', 
    "GuHCl": '(M)', 
    "NaCl": '(M)',
    "KCl": '(M)'
}

# for use with curve_fit
def linearFunc(x,intercept,slope):
    y = intercept + slope * x
    return y

def get_gs_implied_R_e(N, sol, conc):
    linkers = ['GS0','GS8','GS16','GS24','GS32','GS48']
    linker_Ns = [0,16,32,48,64,96]
    linker_R_es = np.array([])
    linker_R_e_errs = np.array([])
    for i in range(len(linkers)):
        R_e = data_df[(data_df.idr == linkers[i]) & (data_df.sol==sol) & (data_df.conc==conc)].R_e_mean
        R_e_err = data_df[(data_df.idr == linkers[i]) & (data_df.sol==sol) & (data_df.conc==conc)].R_e_err
        linker_R_es = np.append(linker_R_es,R_e)
        linker_R_e_errs = np.append(linker_R_e_errs,R_e_err)
    # linear regression for N vs. R_e
    a_fit,cov=curve_fit(linearFunc,linker_Ns,linker_R_es,sigma=linker_R_e_errs)
    linker_R_e_intercept = a_fit[0]
    linker_R_e_slope = a_fit[1]
    return (linker_R_e_intercept + (N * linker_R_e_slope))
    

#####################################################################################
# plot_R_e_heatmap: plots chi vs. concentration for all specified solutes for one IDR
#
# Arguments:
# - idr: the IDR
# - row: the row of the big heatmap where we plot this IDR
# - df: dataframe containing chi values to plot
# - repeats: number of experimental repeats done for this IDR
# - solutes: list of solutes -- each solute is one column of the heatmap
# - titles: column titles representing the solutes
#####################################################################################
def plot_R_e_heatmap_with_gs_lines(idr, idr_title, top_lim, low_lim, row, df, solutes, titles, idr_set):
    N = idr_to_N[idr]
    for i in range(len(solutes)):
        # Plot the cell for this IDR and solute
        col = solutes.index(solutes[i])
        concs = df[(df.idr==idr) & (df.sol==solutes[i])].conc
        R_es = df[(df.idr==idr) & (df.sol==solutes[i])].R_e_mean
        R_e_errs = df[(df.idr==idr) & (df.sol==solutes[i])].R_e_err
        if idr_set != 'gs':
            gs_implied_R_es = np.array([])
            for conc in concs:
                gs_implied_R_es = np.append(gs_implied_R_es,get_gs_implied_R_e(N, solutes[i], conc))
            axs[row][col].plot(concs, gs_implied_R_es,color='black',zorder=100,linestyle='dashed',lw=3)
        axs[row][col].scatter(concs,R_es,c='white',edgecolors='black',linewidths=2,s=350,zorder=50)
        axs[row][col].errorbar(concs,R_es,R_e_errs,fmt='none',capsize=4,capthick=1,ls='none',color='darkgray',elinewidth=1, zorder=100)
        if IDR_SET in ['idrs', 'pumas']:
            if solutes[i] in ['NaCl','KCl']:
                axs[row][col].axhline(get_gs_implied_R_e(N, 'Buffer (no NaCl)', 0), color='black', lw=1, alpha=1, linestyle='dashed')
            else:
                axs[row][col].axhline(get_gs_implied_R_e(N, 'Buffer (standard)', 0), color='black', lw=1, alpha=1, linestyle='dashed')
        if solutes[i] not in ['NaCl','KCl']:
            # Blue-white-red
            cspace=np.linspace(minl,maxl,30)
            location = np.argmin(np.abs(cspace))
            R=np.hstack([np.ones(location), np.linspace(1,0,30-location)])
            G=np.hstack([np.linspace(0,1,location),np.linspace(1,0,30-location)])
            B=np.hstack([np.linspace(0,1,location),np.ones(30-location)])
            cmap=np.vstack([R,G,B]).T
            polyfit_x = concs.astype(float)
            a=np.polyfit(polyfit_x,R_es,1)
            lloc = np.argmin(np.abs(cspace-a[0]*2*np.max(polyfit_x)));
            axs[row][col].set_facecolor(cmap[lloc,:])
        else:
            # White to purple
            R=np.linspace(0.7,1,30)
            G=np.linspace(0,1,30)
            B=np.linspace(1,1,30)
            cmap=np.vstack([R,G,B]).T
            a=np.polyfit(polyfit_x,R_es,2)
            if IDR_SET == 'idrs':
                most_negative_coeff = -11
            elif IDR_SET == 'pumas':
                most_negative_coeff = -9
            else:
                most_negative_coeff = -9
            lloc = 30 - int(a[0] * (30 / most_negative_coeff))
            axs[row][col].set_facecolor(cmap[lloc,:])

        # Set ticks
        top_conc = concs.max()
        margin = top_conc * 0.125
        left_lim = -margin
        right_lim = top_conc + margin
        axs[row][col].set_xlim([left_lim, right_lim])
        if top_conc < 1:
            axs[row][col].set_xticks([0, 0.5])
            axs[row][col].set_xticklabels(['0', '0.5'])
        elif top_conc < 2:
            axs[row][col].set_xticks([0, 1])
            axs[row][col].set_xticklabels(['0', '1'])
        elif top_conc < 3:
            axs[row][col].set_xticks([0, 2])
            axs[row][col].set_xticklabels(['0', '2'])
        elif top_conc < 6:
            axs[row][col].set_xticks([0, 4])
            axs[row][col].set_xticklabels(['0', '4'])
            
        axs[row][col].set_ylim([low_lim, top_lim])
        if row == 0:
            axs[row][col].text(0.5,1.1,titles[i],horizontalalignment='center',
                               fontsize=38,rotation=0,transform=axs[row][col].transAxes)
        if col != 0:
            axs[row][col].yaxis.set_ticks_position('none')
        if col == 0:
            axs[row][col].set_ylabel('$R^{app}_e$ ($\AA$)', fontsize=38, rotation=90)
            axs[row][col].yaxis.set_label_coords(-0.37,0.5)
            axs[row][col].text(-0.8,.5,idr_title,verticalalignment='center',
                               fontsize=45,ha='right',transform=axs[row][col].transAxes)
    return

###################################################################################
# Solute and title lists (title list allows shorter names or nicknames for solutes)
###################################################################################
solutes = ['Ficoll', 'PEG2000', 'Glycine', 'Sarcosine', 'Urea', 'GuHCl', 'NaCl', 'KCl']
titles = ['Ficoll', 'PEG2000', 'Glycine', 'Sarcosine', 'Urea', 'GuHCl', 'NaCl', 'KCl']

if IDR_SET == 'idrs':
    idrs = ['PUMA WT','p53', 'Ash1', 'E1A', 'FUS']
    idr_titles = ['PUMA','p53', 'Ash1', 'E1A', 'FUS']
    top_lims = [90,90,90,90,90]
    low_lims = [36,36,36,36,36]
elif IDR_SET == 'gs':
    idrs = ['GS0','GS8','GS16','GS24','GS32','GS48']
    idr_titles = ['GS0','GS8','GS16','GS24','GS32','GS48']
    top_lims = [87,87,87,87,87,87]
    low_lims = [49,49,49,49,49,49]
elif IDR_SET == 'pumas':
    idrs = ['PUMA WT','PUMA S1','PUMA S2','PUMA S3']
    idr_titles = ['WT','S1','S2','S3']
    top_lims = [77,77,77,77]
    low_lims = [36,36,36,36]

# Create and format the figure
ncols = len(solutes)
nrows = len(idrs)
wspace=0.4
hspace=0.4
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=False, sharey=False, 
                        figsize=(ncols*2.5 + (wspace*(ncols-1)), nrows*2.5 + (hspace*(nrows-1))))
plt.subplots_adjust(wspace=0.04, hspace=0.04)

if IDR_SET == 'gs':
    fig.text(0.51, 0.01, "[solute]", fontsize=40, ha='center')
elif IDR_SET == 'idrs':
    fig.text(0.51, -0.01, "[solute]", fontsize=40, ha='center')
elif IDR_SET == 'pumas':
    fig.text(0.51, -0.04, "[solute]", fontsize=40, ha='center')

for row in range(len(idrs)):
    axs[row][0].yaxis.tick_left()
    
for col in range(ncols):
    axs[nrows-1][col].xaxis.tick_bottom()
    axs[nrows-1][col].set_xlabel(solute_to_axis_label[solutes[col]], fontsize=38)
    
# Build data frame containing chi values for all IDRs in all solution conditions
data_df = pd.read_csv('hidden_structure_fret_and_saxs_aggregated_data.csv')

# Populate the cells in the figure
for i in range(len(idrs)):
    plot_R_e_heatmap_with_gs_lines(idrs[i], idr_titles[i], top_lims[i], low_lims[i], i, data_df, solutes, titles, IDR_SET)

# More formatting
for row in range(nrows):
    for col in range (ncols):
        axs[row][col].label_outer()
        axs[row][col].tick_params(direction='in', length=12, width=5)

# Save the figure
if IDR_SET == 'idrs':
    fig.savefig('hidden_structure_fig_2g.png', bbox_inches='tight')
elif IDR_SET == 'pumas':
    fig.savefig('hidden_structure_fig_3f.png', bbox_inches='tight')
elif IDR_SET == 'gs':
    fig.savefig('hidden_structure_fig_s5.png', bbox_inches='tight')

plt.show()
