########################################################################################################
# hidden_structure_fig_1d.py
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
# 1D is done in this script.
#
# Required files:
# - hidden_structure_fret_and_saxs_aggregated_data.csv
# - hidden_structure_sims_rg_gs_median_q25_q75.csv
# - hidden_structure_sims_re_gs_median_q25_q75.csv
########################################################################################################

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

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

# Create and format the figure
ncols = 1
nrows = 2
wspace = 0
hspace = 0
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=False, figsize=(10.5,10))
plt.subplots_adjust(wspace=wspace, hspace=hspace)
plt.margins(x=0,y=0)

###############
# Fig 1D
###############

exp_data_df = pd.read_csv('hidden_structure_fret_and_saxs_aggregated_data.csv')
linkers = ['GS0','GS8','GS16','GS24','GS32','GS48']
linker_Ns = [0, 16, 32, 48, 64, 96]
linker_colors = ['black','dimgrey','grey','darkgrey','silver','lightgray']
alphas = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75]
for i in range(len(linkers)):
    R_e = exp_data_df[(exp_data_df.idr == linkers[i]) & (exp_data_df.sol=='Buffer (standard)')].R_e_mean
    R_e_err = exp_data_df[(exp_data_df.idr == linkers[i]) & (exp_data_df.sol=='Buffer (standard)')].R_e_err
    R_g = exp_data_df[(exp_data_df.idr == linkers[i]) & (exp_data_df.sol=='Buffer (standard)')].R_g_mean
    R_g_err = exp_data_df[(exp_data_df.idr == linkers[i]) & (exp_data_df.sol=='Buffer (standard)')].R_g_err
    axs[0].scatter(linker_Ns[i], R_g, s=1800, marker='o', color=linker_colors[i], alpha=alphas[i],
                   edgecolors='black', linewidth=3, label='$R_g^{exp}$', zorder=50)
    axs[1].scatter(linker_Ns[i], R_e, s=1800, marker='o', color=linker_colors[i], alpha=alphas[i],
                   edgecolors='black', linewidth=3, label='$R_e^{exp}$', zorder=50)
    axs[0].errorbar(linker_Ns[i], R_g, yerr=R_g_err, capsize=7, capthick=2, ls='none', color='white', 
                    elinewidth=3, zorder=100)
    axs[1].errorbar(linker_Ns[i], R_e, yerr=R_e_err, capsize=7, capthick=2, ls='none', color='white', 
                    elinewidth=3, zorder=100)

sim_R_g_df = pd.read_csv('hidden_structure_sims_rg_gs_median_q25_q75.csv', names=['N', 'R_g', 'q1', 'q3'])
sim_R_e_df = pd.read_csv('hidden_structure_sims_re_gs_median_q25_q75.csv', names=['N', 'R_e', 'q1', 'q3'])
axs[0].scatter(sim_R_g_df['N'] * 2, sim_R_g_df['R_g'], s=800, marker='o', color='blue', 
               alpha = 0.5, label='$R_g^{sim}$', zorder=75, edgecolors='white', linewidth=2)
axs[1].scatter(sim_R_e_df['N'] * 2, sim_R_e_df['R_e'], s=800, marker='o', color='blue', 
               alpha = 0.5, label='$R_e^{sim}$', zorder=75, edgecolors='white', linewidth=2)
axs[0].fill_between(sim_R_g_df.N * 2, sim_R_g_df.q1, sim_R_g_df.q3, color='blue', alpha=0.1)
axs[1].fill_between(sim_R_e_df.N * 2, sim_R_e_df.q1, sim_R_e_df.q3, color='blue', alpha=0.1)

axs[1].set_xlabel('N$_{residues}$', labelpad=-8)
axs[0].set_ylabel('$R_g$ $(\AA)$')
axs[1].set_ylabel('$R^{app}_e$ $(\AA)$')
for ax in axs:
    handles, labels = ax.get_legend_handles_labels()
    ax.legend([handles[0],handles[6]], [labels[0],labels[6]], loc='upper left', fontsize=26, handletextpad=0.1, frameon=False)
    ax.set_xticks([0, 50, 100])
    ax.set_xlim([-6, 102])
fig.tight_layout()
plt.savefig('hidden_structure_fig_1d.png', bbox_inches='tight')
plt.show()

