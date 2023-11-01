import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from scipy.optimize import curve_fit, minimize_scalar

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

# for use in finding peak maximum
def gaussian(x, p1, p2, p3):
    return p3*(p1/((x-p2)**2 + (p1/2)**2))   

# for use with curve_fit
def linearFunc(x,intercept,slope):
    y = intercept + slope * x
    return y

# Create and format the figure
ncols=3
nrows=1
wspace=0.4
fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=False, sharey=False, 
                        figsize=(ncols*12 + (wspace*(ncols-1)), nrows*10))
plt.subplots_adjust(wspace=wspace)
plt.margins(x=0,y=0)

data_df = pd.read_csv('structural_bias_nsmb_in_vitro_data.csv')

###############
# Fig 4A, 4C
###############
# N vs R_g or E_f for idrs
idrs = ['PUMA WT','Ash1', 'E1A', 'FUS', 'p53']
idr_titles = ['PUMA', 'Ash1', 'E1A', 'FUS', 'p53']
idr_Ns = [34, 83, 40, 163, 61]
idr_colors = ['darkslateblue','gold','teal','darkorange','royalblue']
fig4a_df = pd.DataFrame(columns=['idr','N','E_f','E_f_err'])
fig4c_df = pd.DataFrame(columns=['idr','N','R_g','R_g_err'])

for i in range(len(idrs)):
    R_g = data_df[(data_df.idr == idrs[i]) & (data_df.sol=='Buffer (standard)')].guinier_R_g_mean
    R_g_err = data_df[(data_df.idr == idrs[i]) & (data_df.sol=='Buffer (standard)')].guinier_R_g_err
    axs[2].scatter(idr_Ns[i],R_g,s=2500,marker='o',c=idr_colors[i],edgecolors='black', linewidth=3,label=idr_titles[i],alpha=1,zorder=75)
    axs[2].errorbar(idr_Ns[i],R_g,yerr=R_g_err,capsize=4,capthick=2,ls='none',color='silver',elinewidth=3, zorder=100)
    E_f = data_df[(data_df.idr == idrs[i]) & (data_df.sol=='Buffer (standard)')].E_f_mean
    E_f_err = data_df[(data_df.idr == idrs[i]) & (data_df.sol=='Buffer (standard)')].E_f_err
    axs[0].scatter(idr_Ns[i],E_f,s=2500,marker='o',c=idr_colors[i],edgecolors='black', linewidth=3,label=idr_titles[i],alpha=1,zorder=75)
    axs[0].errorbar(idr_Ns[i],E_f,yerr=E_f_err,capsize=4,capthick=2,ls='none',color='silver',elinewidth=3, zorder=100)
    fig4a_df = fig4a_df.append({'idr':idrs[i],'N':idr_Ns[i],'E_f':E_f.values[0],'E_f_err':E_f_err.values[0]}, ignore_index=True)
    fig4c_df = fig4c_df.append({'idr':idrs[i],'N':idr_Ns[i],'R_g':R_g.values[0],'R_g_err':R_g_err.values[0]}, ignore_index=True)

# N vs R_g or E_f regression lines for GS linkers
linkers = ['GS0','GS8','GS16','GS24','GS32','GS48']
linker_Ns = [0, 16, 32, 48, 64, 96]
linker_E_fs = np.array([])
linker_R_gs = np.array([])
linker_E_f_errs = np.array([])
linker_R_g_errs = np.array([])
for i in range(len(linkers)):
    E_f = data_df[(data_df.idr == linkers[i]) & (data_df.sol=='Buffer (standard)')].E_f_mean
    E_f_err = data_df[(data_df.idr == linkers[i]) & (data_df.sol=='Buffer (standard)')].E_f_err
    R_g = data_df[(data_df.idr == linkers[i]) & (data_df.sol=='Buffer (standard)')].guinier_R_g_mean
    R_g_err = data_df[(data_df.idr == linkers[i]) & (data_df.sol=='Buffer (standard)')].guinier_R_g_err
    linker_E_fs = np.append(linker_E_fs,E_f)
    linker_R_gs = np.append(linker_R_gs,R_g)
    linker_E_f_errs = np.append(linker_E_f_errs,E_f_err)
    linker_R_g_errs = np.append(linker_R_g_errs,R_g_err)
    fig4a_df = fig4a_df.append({'idr':linkers[i],'N':linker_Ns[i],'E_f':E_f.values[0],'E_f_err':E_f_err.values[0]}, ignore_index=True)
    fig4c_df = fig4c_df.append({'idr':linkers[i],'N':linker_Ns[i],'R_g':R_g.values[0],'R_g_err':R_g_err.values[0]}, ignore_index=True)
    
# fig4a_df.to_csv('structural-bias-nsmb-fig-4a-n-vs-efret-idps.csv', index=False)
# fig4c_df.to_csv('structural-bias-nsmb-fig-4c-n-vs-rg-idps.csv', index=False)

# Linear fit for N vs. E_f
a_fit,cov=curve_fit(linearFunc,linker_Ns,linker_E_fs,sigma=linker_E_f_errs)
linker_E_f_intercept = a_fit[0]
linker_E_f_slope = a_fit[1]
perr = np.sqrt(np.diag(cov))
linker_E_f_intercept_err = perr[0]
linker_E_f_slope_err = perr[1]
# Linear fit for N vs. R_g
a_fit,cov=curve_fit(linearFunc,linker_Ns,linker_R_gs,sigma=linker_R_g_errs)
linker_R_g_intercept = a_fit[0]
linker_R_g_slope = a_fit[1]
perr = np.sqrt(np.diag(cov))
linker_R_g_intercept_err = perr[0]
linker_R_g_slope_err = perr[1]
# N vs R_g line
axs[2].set_xlim(25,172)
linker_R_g_x_vals = np.array(axs[2].get_xlim())
linker_R_g_y_vals = linker_R_g_intercept + (linker_R_g_slope * linker_R_g_x_vals)
axs[2].plot(linker_R_g_x_vals, linker_R_g_y_vals, c='cadetblue', linestyle='dashed', linewidth=4, zorder=50, label='GS(N)')
linker_R_g_y_upper = (linker_R_g_intercept + linker_R_g_intercept_err) + ((linker_R_g_slope + linker_R_g_slope_err) * linker_R_g_x_vals)
linker_R_g_y_lower = (linker_R_g_intercept - linker_R_g_intercept_err) + ((linker_R_g_slope - linker_R_g_slope_err) * linker_R_g_x_vals)
axs[2].fill_between(linker_R_g_x_vals, linker_R_g_y_lower, linker_R_g_y_upper, color='cadetblue', alpha=0.4)
# N vs E_f line
axs[0].set_xlim(25,172)
linker_E_f_x_vals = np.array(axs[0].get_xlim())
linker_E_f_y_vals = linker_E_f_intercept + (linker_E_f_slope * linker_E_f_x_vals)
axs[0].plot(linker_E_f_x_vals, linker_E_f_y_vals, c='cadetblue', linestyle='dashed', linewidth=4, zorder=50, label='GS(N)')
linker_E_f_y_upper = (linker_E_f_intercept + linker_E_f_intercept_err) + ((linker_E_f_slope + linker_E_f_slope_err) * linker_E_f_x_vals)
linker_E_f_y_lower = (linker_E_f_intercept - linker_E_f_intercept_err) + ((linker_E_f_slope - linker_E_f_slope_err) * linker_E_f_x_vals)
axs[0].fill_between(linker_E_f_x_vals, linker_E_f_y_lower, linker_E_f_y_upper, color='cadetblue', alpha=0.4)

axs[0].set_xlabel('$N_{residues}$')
axs[0].grid(visible=True)
axs[2].set_xlabel('$N_{residues}$')
axs[2].grid(visible=True)
# axs[2].legend(loc='upper left', prop={'size': 28}, handletextpad=0.3, frameon=False, borderpad=0.2, markerscale=0.7)
axs[0].legend(loc='upper right', prop={'size': 23}, handletextpad=0.3, frameon=False, borderpad=0.2, markerscale=0.55)
axs[2].set_ylim(25,54)
axs[2].set_ylabel('$R_g$ $(\AA)$')
axs[0].set_ylabel('$E^{app}_f$')
axs[0].set_ylim(0,0.8)

###############
# Fig. 4B
###############
Ns = np.array([34,61,163,40,83])
csvs = ['20211009_SS07_puma_wt_pbs_chromatogram_for_graphing.csv',
        '20211009_SS08_p53_pbs_chromatogram_for_graphing.csv',
        '20211009_SS17_fus_pbs_chromatogram_for_graphing.csv',
        '20211009_SS10_e1a_pbs_chromatogram_for_graphing.csv',
        '20211009_SS09_ash1_pbs_chromatogram_for_graphing.csv']
labels = ['PUMA', 'p53', 'FUS', 'E1A', 'Ash1']
colors = ['darkslateblue','royalblue','darkorange','teal','gold']
peak_positions = np.array([])
peak_position_errs = np.array([])
fig4b_df = pd.DataFrame(columns=['idr','N','peak_position','peak_position_err'])

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
    fig4b_df = fig4b_df.append({'idr':labels[i],'N':Ns[i],'peak_position':peak_position,'peak_position_err':peak_position_err}, ignore_index=True)
axs[1].scatter(Ns, peak_positions, c=colors, edgecolors='black', linewidth=3, label=labels[i], s=2500, zorder=75)
axs[1].errorbar(Ns, peak_positions, yerr=peak_position_errs, capsize=4, ls='none', color='silver', elinewidth=3, zorder=100)
    
axs[1].xaxis.tick_bottom()
axs[1].yaxis.tick_left()
axs[1].set_xlim([23, 174])
axs[1].set_xlabel('$N_{residues}$')
axs[1].set_ylabel('elution volume (mL)')
    
# Plot line for GS linkers
gs_names = ['GS0','GS8','GS24','GS32','GS48']
gs_Ns = np.array([0,16,48,64,96])
gs_csvs = ['20211009_SS1_gs0_pbs_chromatogram_for_graphing.csv',
        '20211009_SS2_gs8_pbs_chromatogram_for_graphing.csv',
        '20211009_SS04_gs24_pbs_chromatogram_for_graphing.csv',
        '20211009_SS05_gs32_pbs_chromatogram_for_graphing.csv',
        '20211009_SS06_gs48_pbs_chromatogram_for_graphing.csv']
gs_peak_positions = np.array([])
gs_peak_position_errs = np.array([])

for i in range(len(gs_csvs)):
    df = pd.read_csv(gs_csvs[i])
    gs_frame = df.Frame.to_numpy()
    # Exposure period is 2 seconds. Flow rate is 0.6 mL/min. Error is +/- 2 frames.
    gs_volumes = gs_frame * 2 * (0.6 / 60)
    gs_intensity = df.Integrated_Intensity.to_numpy()
    # First estimate of peak location: index with max intensity value
    gs_peak_index = np.argmax(gs_intensity)
    # Find true peak maximum
    popt, pcov = curve_fit(gaussian, gs_volumes[gs_peak_index-10:gs_peak_index+10], 
                            gs_intensity[gs_peak_index-10:gs_peak_index+10], 
                            p0=(1,gs_peak_index*2 *(0.6/60),1))
    fm = lambda x: -gaussian(x, *popt)
    r = minimize_scalar(fm, bounds=(1, 5))
    gs_peak_position = r["x"]
    # We assume an error of one frame in each direction.
    gs_peak_position_err = 2 * (0.6 / 60) # volume in mL of one frame
    gs_peak_positions = np.append(gs_peak_positions, gs_peak_position)
    gs_peak_position_errs = np.append(gs_peak_position_errs, gs_peak_position_err)
    fig4b_df = fig4b_df.append({'idr':gs_names[i],'N':Ns[i],'peak_position':gs_peak_position,'peak_position_err':gs_peak_position_err}, ignore_index=True)
    
# fig4b_df.to_csv('structural-bias-nsmb-fig-4b-n-vs-sec-peak-position-idps.csv', index=False)
    
# Linear fit for N vs. peak position
a_fit,cov=curve_fit(linearFunc,gs_Ns,gs_peak_positions,sigma=gs_peak_position_errs)
gs_peak_position_intercept = a_fit[0]
gs_peak_position_slope = a_fit[1]
perr = np.sqrt(np.diag(cov))
gs_peak_position_intercept_err = perr[0]
gs_peak_position_slope_err = perr[1]
x_vals = np.array(axs[1].get_xlim())
y_vals = gs_peak_position_intercept + (gs_peak_position_slope * x_vals)
axs[1].plot(x_vals, y_vals, c='cadetblue', linestyle='dashed', linewidth=4, zorder=0, label='GS(N)')
y_upper = (gs_peak_position_intercept + gs_peak_position_intercept_err) + ((gs_peak_position_slope + gs_peak_position_slope_err) * x_vals)
y_lower = (gs_peak_position_intercept - gs_peak_position_intercept_err) + ((gs_peak_position_slope - gs_peak_position_slope_err) * x_vals)
axs[1].fill_between(x_vals, y_lower, y_upper, color='cadetblue', alpha=0.4)

axs[1].grid(visible=True)

# fig.tight_layout()
plt.savefig('structural_bias_nsmb_fig_4A_4B_4C.png', bbox_inches='tight')
plt.show()

