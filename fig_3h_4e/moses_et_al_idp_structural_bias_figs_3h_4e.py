######################################################################################################################
#
#
#
######################################################################################################################


import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit

mpl.style.use('default')
plt.rc('xtick', labelsize=40) 
plt.rc('ytick', labelsize=40) 
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
# IDR_SET = 'idrs' # Uncomment for Fig 4E
IDR_SET = 'pumas' # Uncomment for Fig 3H

###########################
# Make colormap for heatmap
###########################
# range of slopes for color map
min_delta=-0.8
max_delta=0.8

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

def quadraticFunc(x, a, b, c):
	return a * x**2 + b * x + c

def get_gs_implied_E_f(N, sol, conc):
    linkers = ['GS0','GS8','GS16','GS24','GS32','GS48']
    linker_Ns = [0,16,32,48,64,96]
    linker_E_fs = np.array([])
    linker_E_f_errs = np.array([])
    for i in range(len(linkers)):
        E_f = data_df[(data_df.idr == linkers[i]) & (data_df.sol==sol) & (data_df.conc==conc)].E_f_mean
        E_f_err = data_df[(data_df.idr == linkers[i]) & (data_df.sol==sol) & (data_df.conc==conc)].E_f_err
        linker_E_fs = np.append(linker_E_fs,E_f)
        linker_E_f_errs = np.append(linker_E_f_errs,E_f_err)
    # linear regression for N vs. E_f
    # error for interpolated E_f or R_g will be sqrt(N**2 * slope_err**2 + intercept_err**2)
    # one reference for this is www.itl.nist.gov/div898/handbook/mpc/section5/mpc552.htm
    a_fit,cov=curve_fit(linearFunc,linker_Ns,linker_E_fs,sigma=linker_E_f_errs)
    linker_E_f_intercept = a_fit[0]
    linker_E_f_slope = a_fit[1]
    linker_E_f_intercept_err = np.sqrt(cov[0][0])
    linker_E_f_slope_err = np.sqrt(cov[1][1])
    gs_implied_E_f = (linker_E_f_intercept + (N * linker_E_f_slope))
    gs_implied_E_f_err = np.sqrt((N**2 * linker_E_f_slope_err**2) + linker_E_f_intercept_err**2)
    return (gs_implied_E_f, gs_implied_E_f_err)

def get_gs_implied_E_f_quadratic(N, sol, conc):
    linkers = ['GS0','GS8','GS16','GS24','GS32','GS48']
    linker_Ns = [0,16,32,48,64,96]
    linker_E_fs = np.array([])
    linker_E_f_errs = np.array([])
    for i in range(len(linkers)):
        E_f = data_df[(data_df.idr == linkers[i]) & (data_df.sol==sol) & (data_df.conc==conc)].E_f_mean
        E_f_err = data_df[(data_df.idr == linkers[i]) & (data_df.sol==sol) & (data_df.conc==conc)].E_f_err
        linker_E_fs = np.append(linker_E_fs,E_f)
        linker_E_f_errs = np.append(linker_E_f_errs,E_f_err)
    # linear regression for N vs. E_f
    # error for interpolated E_f or R_g will be sqrt(N**2 * slope_err**2 + intercept_err**2)
    # one reference for this is www.itl.nist.gov/div898/handbook/mpc/section5/mpc552.htm
    popt,cov=curve_fit(quadraticFunc,linker_Ns,linker_E_fs,sigma=linker_E_f_errs)
    a, b, c = popt
    # print('y = %.5f * x + %.5f * x^2 + %.5f' % (a, b, c))
    a_err, b_err, c_err = np.sqrt(np.diag(cov))
    gs_implied_E_f = a * N**2 + b * N + c
    # gs_implied_E_f_err = np.sqrt((N**2 * linker_E_f_slope_err**2) + linker_E_f_intercept_err**2)
    return (gs_implied_E_f)
    

#####################################################################################
# plot_E_f_heatmap: plots chi vs. concentration for all specified solutes for one IDR
#
# Arguments:
# - idr: the IDR
# - row: the row of the big heatmap where we plot this IDR
# - df: dataframe containing chi values to plot
# - repeats: number of experimental repeats done for this IDR
# - solutes: list of solutes -- each solute is one column of the heatmap
# - titles: column titles representing the solutes
#####################################################################################
def plot_E_f_heatmap_normalize_to_gs(idr, idr_title, top_lim, low_lim, row, df, solutes, titles, idr_set):
    N = idr_to_N[idr]
    for i in range(len(solutes)):
        # Plot the cell for this IDR and solute
        col = solutes.index(solutes[i])
        concs = df[(df.idr==idr) & (df.sol==solutes[i])].conc
        E_f_naught = df[(df.idr==idr) & (df.sol==solutes[i]) & (df.conc==0)].E_f_mean.values[0]
        E_f_naught_err = df[(df.idr==idr) & (df.sol==solutes[i]) & (df.conc==0)].E_f_err.values[0]
        gs_implied_E_f_naught, gs_implied_E_f_naught_err = get_gs_implied_E_f(N, solutes[i], 0)
        E_fs = np.array([])
        gs_implied_E_fs = np.array([])
        for conc in concs:
            E_f = df[(df.idr==idr) & (df.sol==solutes[i]) & (df.conc==conc)].E_f_mean.values[0]
            E_fs = np.append(E_fs, E_f)
            E_f_err = df[(df.idr==idr) & (df.sol==solutes[i]) & (df.conc==conc)].E_f_err.values[0]
            gs_implied_E_f = get_gs_implied_E_f_quadratic(N, solutes[i], conc)
            gs_implied_E_fs = np.append(gs_implied_E_fs, gs_implied_E_f)
            if conc == 0:
                axs[row][col].scatter(0, 0, c='white', edgecolors='black', linewidths=2, s=350, zorder=50)
                axs[row][col].errorbar(0, 0, yerr=E_f_naught_err, capsize=4, capthick=2, ls='none', 
                                       color='darkgray', elinewidth=3, zorder=100)
            else:
                axs[row][col].scatter(conc, E_f - E_f_naught, c='white', edgecolors='black', 
                                      linewidths=2, s=350, zorder=50)
                axs[row][col].errorbar(conc, E_f - E_f_naught, yerr=np.sqrt(E_f_err**2 + E_f_naught_err**2), 
                                       capsize=4, capthick=2, ls='none', color='darkgray', elinewidth=3, zorder=100)
        if idr != 'FUS':
            axs[row][col].plot(concs, gs_implied_E_fs - gs_implied_E_f_naught, color='black', zorder=100, linestyle='dashed', lw=4)
            axs[row][col].fill_between(concs, E_fs - E_f_naught, gs_implied_E_fs - gs_implied_E_f_naught, color='cadetblue', alpha=1)
        if solutes[i] in ['NaCl', 'KCl']:
            delta_E_fs = E_fs[2:] - E_fs[2]
            delta_gs_implied_E_fs = gs_implied_E_fs[2:] - gs_implied_E_fs[2]
        else:
            delta_E_fs = E_fs - E_f_naught
            delta_gs_implied_E_fs = gs_implied_E_fs - gs_implied_E_f_naught
        relative_sensitivity = abs(sum(delta_E_fs) - sum(delta_gs_implied_E_fs))
        if delta_E_fs[-1] > 0: # E_f increases, so compaction
            if (sum(delta_E_fs) < sum(delta_gs_implied_E_fs)): # less compaction, so less sensitive than GS
                relative_sensitivity = -relative_sensitivity
        else: # E_f decreases, so expansion
            if (sum(delta_E_fs) > sum(delta_gs_implied_E_fs)): # less expansion, so less sensitive than GS
                relative_sensitivity = -relative_sensitivity
            
        # Blue-white-red colormap
        cspace=np.linspace(min_delta,max_delta,30)
        midpoint = np.argmin(np.abs(cspace))
        B=np.hstack([np.ones(midpoint), np.linspace(1,0,30-midpoint)])
        G=np.hstack([np.linspace(0,1,midpoint),np.linspace(1,0,30-midpoint)])
        R=np.hstack([np.linspace(0,1,midpoint),np.ones(30-midpoint)])
        cmap=np.vstack([R,G,B]).T
        cmapindex = midpoint + int(midpoint*relative_sensitivity/max_delta)
        if cmapindex>29:
            cmapindex=29
        # To avoid unreliable extrapolation, color FUS white and don't compare it with GS.
        if idr == 'FUS':
            axs[row][col].set_facecolor('white')
        else:
            axs[row][col].set_facecolor(cmap[cmapindex,:])
        if solutes[i] in ['NaCl', 'KCl']:
            axs[row][col].axvspan(-0.14, 0.300, facecolor='lightgray', alpha=0.5)

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
            
        # Draw vertical line for salts to separate screening
        if solutes[i] in ['NaCl', 'KCl']:
            axs[row][col].axvline(x=0.300, c='black', linewidth=2, linestyle=':')
            
        axs[row][col].set_ylim([low_lim, top_lim])
        if row == 0:
            axs[row][col].text(0.5,1.1,titles[i],horizontalalignment='center',
                               fontsize=38,rotation=0,transform=axs[row][col].transAxes)
        if col != 0:
            axs[row][col].yaxis.set_ticks_position('none')
        if col == 0:
            if idr_set == 'idrs':
                axs[row][col].text(-0.42,.5,idr_title,verticalalignment='center',fontsize=60,ha='right',transform=axs[row][col].transAxes)
            if idr_set == 'pumas':
                axs[row][col].text(-0.45,.5,idr_title,verticalalignment='center',fontsize=60,ha='right',transform=axs[row][col].transAxes)

    return

###################################################################################
# Solute and title lists (title list allows shorter names or nicknames for solutes)
###################################################################################
solutes = ['Ficoll', 'PEG2000', 'Glycine', 'Sarcosine', 'Urea', 'GuHCl', 'NaCl', 'KCl']
titles = ['Ficoll', 'PEG2000', 'Glycine', 'Sarcosine', 'Urea', 'GuHCl', 'NaCl', 'KCl']

if IDR_SET == 'idrs':
    idrs = ['PUMA WT', 'Ash1', 'E1A', 'FUS', 'p53']
    idr_titles = ['PUMA','Ash1', 'E1A', 'FUS', 'p53']
    top_lims = [0.3,0.3,0.3,0.3,0.3]
    low_lims = [-0.3,-0.3,-0.3,-0.3,-0.3]
elif IDR_SET == 'pumas':
    idrs = ['PUMA WT','PUMA S1','PUMA S2','PUMA S3']
    idr_titles = ['WT','S1','S2','S3']
    top_lims = [0.3,0.3,0.3,0.3]
    low_lims = [-0.3,-0.3,-0.3,-0.3]

# Create and format the figure
ncols = len(solutes)
nrows = len(idrs)
wspace=0.4
hspace=0.4
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=False, sharey=False, 
                        figsize=(ncols*2.5 + (wspace*(ncols-1)), nrows*2.5 + (hspace*(nrows-1))))
plt.subplots_adjust(wspace=0.04, hspace=0.04)

# Add giant x and y labels for entire figure.
if IDR_SET == 'idrs':
    fig.text(0.51, -0.025, "[solute]", fontsize=58, ha='center')
    plt.text(-0.06, 0.5, '$\Delta$$E^{app}_f$', rotation=90, fontsize=75, transform=fig.transFigure, verticalalignment='center')
elif IDR_SET == 'pumas':
    fig.text(0.51, -0.06, "[solute]", fontsize=58, ha='center')
    plt.text(-0.02, 0.5, '$\Delta$$E^{app}_f$', rotation=90, fontsize=70, transform=fig.transFigure, verticalalignment='center')

for row in range(len(idrs)):
    axs[row][0].yaxis.tick_left()
    
for col in range(ncols):
    axs[nrows-1][col].xaxis.tick_bottom()
    axs[nrows-1][col].set_xlabel(solute_to_axis_label[solutes[col]], fontsize=38)
    
# Build data frame containing chi values for all IDRs in all solution conditions
data_df = pd.read_csv('hidden_structure_fret_and_saxs_aggregated_data.csv')

# Populate the cells in the figure
for i in range(len(idrs)):
    plot_E_f_heatmap_normalize_to_gs(idrs[i], idr_titles[i], top_lims[i], low_lims[i], i, data_df, solutes, titles, IDR_SET)

# More formatting
for row in range(nrows):
    if IDR_SET == 'idrs':
        axs[row][0].set_yticks([-0.2,0,0.2])
    elif IDR_SET == 'pumas': 
        axs[row][0].set_yticks([-0.2,0,0.2])
    for col in range (ncols):
        axs[row][col].label_outer()
        axs[row][col].tick_params(direction='in', length=12, width=5)
        axs[row][col].axhline(y=0, linewidth=2, linestyle=':', color='silver')
        
plt.show()

# Save the figure
if IDR_SET == 'pumas':
    fig.savefig('moses_et_al_idp_structural_bias_fig_3h.png', bbox_inches='tight')
    fig.savefig('moses_et_al_idp_structural_bias_fig_3h.svg', bbox_inches='tight')
elif IDR_SET == 'idrs':
    fig.savefig('moses_et_al_idp_structural_bias_fig_4e.png', bbox_inches='tight')
    fig.savefig('moses_et_al_idp_structural_bias_fig_4e.svg', bbox_inches='tight')

