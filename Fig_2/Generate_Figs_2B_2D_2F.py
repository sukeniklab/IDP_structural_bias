import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from scipy.optimize import curve_fit, minimize_scalar
# import scipy.odr.odrpack as odrpack

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

# ax.grid(b=True)

# for use with curve_fit
def linearFunc(x,intercept,slope):
    y = intercept + slope * x
    return y

# for use in finding peak maximum
def gaussian(x, p1, p2, p3):
    return p3*(p1/((x-p2)**2 + (p1/2)**2))   

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
ncols=3
nrows=1
wspace=0.4
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=False, sharey=False, 
                        figsize=(ncols*12 + (wspace*(ncols-1)), nrows*10))
plt.subplots_adjust(wspace=wspace)
plt.margins(x=0,y=0)

exp_data_df = pd.read_csv('structural_bias_nsmb_in_vitro_data.csv')

###############
# Fig 2B
###############

linkers = ['GS0','GS8','GS16','GS24','GS32','GS48']
linker_Ns = np.array([0, 16, 32, 48, 64, 96])
linker_colors = ['black','dimgrey','grey','darkgrey','silver','lightgray']
alphas = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75]
linker_E_fs = np.array([])
linker_E_f_errs = np.array([])
fig2b_df = pd.DataFrame(columns=['idr','N','E_f','E_f_err'])

for i in range(len(linkers)):
    E_f = exp_data_df[(exp_data_df.idr == linkers[i]) & (exp_data_df.sol=='Buffer (standard)')].E_f_mean
    E_f_err = exp_data_df[(exp_data_df.idr == linkers[i]) & (exp_data_df.sol=='Buffer (standard)')].E_f_err
    axs[0].scatter(linker_Ns[i], E_f, s=2500, marker='o', color=linker_colors[i], alpha=alphas[i],
                   edgecolors='black', linewidth=3, label='experiment', zorder=75)
    axs[0].errorbar(linker_Ns[i], E_f, yerr=E_f_err, capsize=4, capthick=2, ls='none', color='white', 
                    elinewidth=3, zorder=100)
    linker_E_fs = np.append(linker_E_fs,E_f)
    linker_E_f_errs = np.append(linker_E_f_errs,E_f_err)
    fig2b_df = fig2b_df.append({'idr':linkers[i],'N':linker_Ns[i],'E_f':E_f.values[0],'E_f_err':E_f_err.values[0]}, ignore_index=True)
    
# fig2b_df.to_csv('structural-bias-nsmb-fig-2b-n-vs-efret-gs.csv', index=False)

# linear regression for N vs. E_f
a_fit,cov=curve_fit(linearFunc,linker_Ns,linker_E_fs,sigma=linker_E_f_errs)
intercept = a_fit[0]
slope = a_fit[1]
perr = np.sqrt(np.diag(cov))
intercept_err = perr[0]
slope_err = perr[1]
axs[0].set_xlim([-10, 170]) # needed for comparison with FUS
x_vals = np.array(axs[0].get_xlim())
y_vals = intercept + (slope * x_vals)
y_upper = (intercept + intercept_err) + ((slope + slope_err) * x_vals)
y_lower = (intercept - intercept_err) + ((slope - slope_err) * x_vals)
axs[0].plot(x_vals, y_vals, c='cadetblue', linestyle='dashed', linewidth=4, zorder=0)
axs[0].fill_between(x_vals, y_lower, y_upper, color='cadetblue', alpha=0.1)

# Convert sim_R_e to sim_E_f
# The equation is E_f = R_naught**6 / (R_naught**6 + R_e**6)
R_naught = 62
sim_R_e_df = pd.read_csv('structural_bias_nsmb_fig_2b_sims_re_gs_median_q25_q75.csv', names=['idr', 'N', 'R_e', 'q1', 'q3'])
sim_idrs = sim_R_e_df.idr
Ns = sim_R_e_df['N'] * 2
E_fs = R_naught**6 / (R_naught**6 + sim_R_e_df['R_e']**6)
E_f_q1s = R_naught**6 / (R_naught**6 + sim_R_e_df.q1**6)
E_f_q3s = R_naught**6 / (R_naught**6 + sim_R_e_df.q3**6)
high_errs = -(E_fs - E_f_q1s)
print (high_errs)
low_errs = -(E_f_q3s - E_fs)
print (low_errs)
axs[0].scatter(Ns, E_fs, s=1100, marker='o', color='darkviolet', alpha = 1, label='simulation', zorder=75, edgecolors='black', linewidth=2)
axs[0].errorbar(Ns, E_fs, yerr=[low_errs, high_errs], capsize=4, capthick=2, ls='none', color='darkviolet', elinewidth=3, zorder=90)

fig2b_sim_df = pd.DataFrame(columns=['idr','N','E_f','low_err','high_err'])
for i in range(len(sim_idrs)):
    fig2b_sim_df = fig2b_sim_df.append({'idr':sim_idrs[i],'N':Ns[i],'E_f':E_fs[i],
                                        'low_err':low_errs[i],
                                        'high_err':high_errs[i]}, ignore_index=True)
# fig2b_sim_df.to_csv('structural-bias-nsmb-fig-2b-n-vs-efret-sims.csv', index=False)

axs[0].set_xlabel('$N_{residues}$')
axs[0].set_ylabel('$E^{app}_f$')
axs[0].set_xticks([0, 50, 100, 150])
axs[0].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
axs[0].set_ylim([0, 1])
h, l = axs[0].get_legend_handles_labels()
handles = [h[0],h[-1]]
labels = [l[0],l[-1]]
axs[0].legend(handles, labels, fontsize=38, frameon=False, loc='upper right', handletextpad=0.1, borderpad=0.1) 
axs[0].grid(visible=True)

###############
# Fig 2D
###############

Ns = np.array([0,16,48,64,96])
csvs = ['20211009_SS1_gs0_pbs_chromatogram_for_graphing.csv',
        '20211009_SS2_gs8_pbs_chromatogram_for_graphing.csv',
        '20211009_SS04_gs24_pbs_chromatogram_for_graphing.csv',
        '20211009_SS05_gs32_pbs_chromatogram_for_graphing.csv',
        '20211009_SS06_gs48_pbs_chromatogram_for_graphing.csv']
labels = ['GS0', 'GS8', 'GS24', 'GS32', 'GS48']  
colors = ['black','dimgrey','darkgrey','silver','lightgray']         
peak_positions = np.array([])
peak_position_errs = np.array([])
fig2d_df = pd.DataFrame(columns=['idr','N','peak_position','peak_position_err'])

for i in range(len(csvs)):
    df = pd.read_csv(csvs[i])
    frame = df.Frame.to_numpy()
    # Exposure period is 2 seconds. Flow rate is 0.6 mL/min. Error is +/- 2 frames.
    volumes = frame * 2 * (0.6 / 60)
    intensity = df.Integrated_Intensity.to_numpy()
    # First estimate of peak location: index with max intensity value
    peak_index = np.argmax(intensity)
    # Find true peak maximum
    popt, pcov = curve_fit(gaussian, volumes[peak_index-10:peak_index+10], intensity[peak_index-10:peak_index+10], p0=(1,peak_index*2 *(0.6/60),1))
    fm = lambda x: -gaussian(x, *popt)
    r = minimize_scalar(fm, bounds=(1, 5))
    # print ("maximum from gaussian:", r["x"], gaussian(r["x"], *popt))
    # peak_position = r["x"] * 2 * (0.6 / 60)
    peak_position = r["x"]
    # Instead of error from fitting the peak, we will assume an error of one frame in each direction.
    # peak_position_err = np.sqrt(np.diag(pcov))[1]
    peak_position_err = 2 * (0.6 / 60) # volume in mL of one frame
    peak_positions = np.append(peak_positions, peak_position)
    peak_position_errs = np.append(peak_position_errs, peak_position_err)
    axs[1].scatter(Ns[i], peak_positions[i], c=colors[i], edgecolors='black', label=labels[i], linewidth=3, s=2500, zorder=75)
    axs[1].errorbar(Ns[i], peak_positions[i], yerr=peak_position_errs[i], ls='none', capsize=4, color='white', elinewidth=3, zorder=100)
    fig2d_df = fig2d_df.append({'idr':labels[i],'N':Ns[i],'peak_position':peak_positions[i],'peak_position_err':peak_position_errs[i]}, ignore_index=True)
    
# fig2d_df.to_csv('structural-bias-nsmb-fig-2d-n-vs-sec-peak-position-gs.csv', index=False)
    
# Linear fit
a_fit,cov=curve_fit(linearFunc,Ns,peak_positions,sigma=peak_position_errs)
intercept = a_fit[0]
slope = a_fit[1]
perr = np.sqrt(np.diag(cov))
intercept_err = perr[0]
slope_err = perr[1]
axs[1].set_xlim([-10, 170]) # needed for comparison with FUS
x_vals = np.array(axs[1].get_xlim())
y_vals = intercept + (slope * x_vals)
y_upper = (intercept + intercept_err) + ((slope + slope_err) * x_vals)
y_lower = (intercept - intercept_err) + ((slope - slope_err) * x_vals)
axs[1].plot(x_vals, y_vals, c='cadetblue', linestyle='dashed', linewidth=4, zorder=0)
axs[1].fill_between(x_vals, y_lower, y_upper, color='cadetblue', alpha=0.1)

residuals = peak_positions - linearFunc(Ns, *a_fit)
ss_res = np.sum(residuals**2)
# total sum of squares
ss_tot = np.sum((peak_positions - np.mean(peak_positions))**2)
r_squared = 1 - (ss_res / ss_tot)

axs[1].set_xticks([0, 50, 100, 150])
axs[1].xaxis.tick_bottom()
axs[1].yaxis.tick_left()
axs[1].set_xlabel('$N_{residues}$')
axs[1].set_ylabel('elution volume (mL)')
axs[1].legend(fontsize=38, loc='lower left', frameon=False, borderpad=0.1, handletextpad=0.1) 
axs[1].grid(visible=True)

###############
# Fig 2F
###############

linker_R_gs = np.array([])
linker_R_g_errs = np.array([])
fig2f_df = pd.DataFrame(columns=['idr','N','R_g','R_g_err'])

for i in range(len(linkers)):
    R_g = exp_data_df[(exp_data_df.idr == linkers[i]) & (exp_data_df.sol=='Buffer (standard)')].guinier_R_g_mean
    R_g_err = exp_data_df[(exp_data_df.idr == linkers[i]) & (exp_data_df.sol=='Buffer (standard)')].guinier_R_g_err
    axs[2].scatter(linker_Ns[i], R_g, s=2500, marker='o', color=linker_colors[i], alpha=alphas[i],
                   edgecolors='black', linewidth=3, label='experiment', zorder=75)
    axs[2].errorbar(linker_Ns[i], R_g, yerr=R_g_err, capsize=4, capthick=2, ls='none', color='white', 
                    elinewidth=3, zorder=100)
    linker_R_gs = np.append(linker_R_gs,R_g)
    linker_R_g_errs = np.append(linker_R_g_errs,R_g_err)
    fig2f_df = fig2f_df.append({'idr':linkers[i],'N':linker_Ns[i],'R_g':R_g.values[0],'R_g_err':R_g_err.values[0]}, ignore_index=True)
    
# fig2f_df.to_csv('structural-bias-nsmb-fig-2f-n-vs-rg-gs.csv', index=False)

# Linear fit for N vs. R_g
a_fit,cov=curve_fit(linearFunc,linker_Ns,linker_R_gs,sigma=linker_R_g_errs)
intercept = a_fit[0]
slope = a_fit[1]
perr = np.sqrt(np.diag(cov))
intercept_err = perr[0]
slope_err = perr[1]
axs[2].set_xlim([-10, 170]) # needed for comparison with FUS
x_vals = np.array(axs[2].get_xlim())
y_vals = intercept + (slope * x_vals)
y_upper = (intercept + intercept_err) + ((slope + slope_err) * x_vals)
y_lower = (intercept - intercept_err) + ((slope - slope_err) * x_vals)
axs[2].plot(x_vals, y_vals, c='cadetblue', linestyle='dashed', linewidth=4, zorder=0)
axs[2].fill_between(x_vals, y_lower, y_upper, color='cadetblue', alpha=0.1)

residuals = linker_R_gs - linearFunc(linker_Ns, *a_fit)
ss_res = np.sum(residuals**2)
# total sum of squares
ss_tot = np.sum((linker_R_gs - np.mean(linker_R_gs))**2)
r_squared = 1 - (ss_res / ss_tot)

sim_R_g_df = pd.read_csv('structural_bias_nsmb_fig_2f_sims_rg_gs_median_q25_q75.csv', names=['idr','N','R_g','q1','q3'])
sim_idrs = sim_R_g_df.idr
Ns = sim_R_g_df.N * 2
R_gs = sim_R_g_df.R_g
R_g_q1s = sim_R_g_df.q1
R_g_q3s = sim_R_g_df.q3
low_errs = R_gs - R_g_q1s
high_errs = R_g_q3s - R_gs
axs[2].scatter(Ns, R_gs, s=1100, marker='o', color='darkviolet', alpha = 1, label='simulation', zorder=75, edgecolors='black', linewidth=2)
axs[2].errorbar(Ns, R_gs, yerr=[low_errs, high_errs], capsize=4, capthick=2, ls='none', color='darkviolet', elinewidth=3, zorder=90)

fig2f_sim_df = pd.DataFrame(columns=['idr','N','R_g','low_err','high_err'])
for i in range(len(sim_idrs)):
    fig2f_sim_df = fig2f_sim_df.append({'idr':sim_idrs[i],'N':Ns[i],'R_g':R_gs[i],'low_err':low_errs[i],'high_err':high_errs[i]}, ignore_index=True)
# fig2f_sim_df.to_csv('structural-bias-nsmb-fig-2f-n-vs-rg-sims.csv', index=False)

axs[2].set_xlabel('$N_{residues}$')
axs[2].set_ylabel('$R_g$ $(\AA)$')

axs[2].set_xticks([0, 50, 100, 150])
axs[2].set_ylim([28.8, 50])
h, l = axs[2].get_legend_handles_labels()
handles = [h[0],h[-1]]
labels = [l[0],l[-1]]
axs[2].legend(handles, labels, fontsize=38, frameon=False, loc='lower right', handletextpad=0.1, borderpad=0.1) 
axs[2].grid(visible=True)
    
# fig.tight_layout()
plt.savefig('structural_bias_nsmb_fig_2B_2D_2F.png', bbox_inches='tight')
plt.show()

