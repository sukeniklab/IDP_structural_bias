########################################################################################################
# Generate_Fig_5B.py
# Generates Fig 5B of "Structural biases in disordered proteins are prevalent in the cell."
# Author: David Moses (dmoses5@ucmerced.edu)
########################################################################################################

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import matplotlib as mpl
import random

mpl.style.use('default')
mpl.rcParams['axes.linewidth'] = 7 #set the value globally
mpl.rcParams['xtick.major.size'] = 20
mpl.rcParams['xtick.major.width'] = 7
mpl.rcParams['xtick.minor.size'] = 10
mpl.rcParams['xtick.minor.width'] = 7
mpl.rcParams['ytick.major.size'] = 20
mpl.rcParams['ytick.major.width'] = 7
mpl.rcParams['ytick.labelsize'] = 45
mpl.rcParams['xtick.labelsize'] = 45
mpl.rcParams['ytick.minor.size'] = 10
mpl.rcParams['ytick.minor.width'] = 7
mpl.rcParams['font.size'] = 55
mpl.rcParams['font.sans-serif']='Arial'

###################################################################################################
# Corrections that need to be made:
# - Area normalization of all spectra
# - Subtraction of emission due to direct acceptor excitation (implied spectrum shape and size)
###################################################################################################

def calculate_monomer_efret(corrected_idr_spectrum, tq_avg_spectrum, ng_avg_spectrum):
    reg = LinearRegression(fit_intercept=False, positive=True)
    Q_d=0.93 # Mastop et al Sci Rep 2017
    Q_a=0.8  # Mastop et al Sci Rep 2017 
            
    # f_d and f_a are corrected and normalized donor and acceptor base spectra
    f_d = tq_avg_spectrum / tq_avg_spectrum[26]
    f_a = ng_avg_spectrum / tq_avg_spectrum[26]
    y = corrected_idr_spectrum
    
    # Join donor and acceptor base spectra side-by-side to send to linear regression function
    X = pd.concat([f_d,f_a],axis=1) 

    # d over a to compare with live cell microscopy results
    d_over_a = sum(y[19:42]) / sum(y[80:])
    
    # Send the experimental spectrum and the base spectra to the linear regression function reg.fit
    # It will return coefficients representing the donor and acceptor contributions to the experimental spectrum
    try:
        reg.fit(X, y)
        # fitResult is the fitted spectrum that is a linear combination of the base spectra
        # fit_result=np.sum(X*reg.coef_,axis=1) 
        # Donor component of corrected experimental spectrum 
        F_d=reg.coef_[0] * f_d
        # Acceptor component of corrected experimental spectrum
        F_s=reg.coef_[1] * f_a
        # Normalize so that area under each curve is 1
        f_d_area_normed = f_d / sum(f_d)
        f_a_area_normed = f_a / sum(f_a)
        # Calculate E_fret - from Mastop 2017, page 15
        E_f_all_lambda = 1 - (F_d / ((((Q_d*f_d_area_normed) / (Q_a*f_a_area_normed)) * F_s) + F_d)) 
        weighting_all_lambda = f_a_area_normed * f_d_area_normed
        E_f = sum(weighting_all_lambda * E_f_all_lambda) / sum(weighting_all_lambda)
        # print ('E_f = ' + str(E_f))
    except:
        print ('calculate_efret: reg fit did not work!')
    return (E_f, d_over_a)

E_f_results_df = pd.DataFrame(columns=['category','idr','E_f'])
graph_df = pd.DataFrame(columns=['category','category_index','idr','idr_index','E_f_mean','E_f_err'])

experiment_data_df = pd.read_csv('fig-5b-fret-data.csv')
raw_base_spectra_df = pd.read_csv('fig-5b-base-spectra-data.csv')
averaged_base_spectra_df = raw_base_spectra_df.groupby(['protein','temp']).mean().reset_index()
avg_dirA_df = pd.read_csv('avg_dirA_df.csv')
avg_idr_dirA_intensity_ratio = avg_dirA_df.avg_idr_dirA_intensity_ratio.values[0]

for index, row in experiment_data_df.iterrows():
    tq_base_spec = averaged_base_spectra_df[(averaged_base_spectra_df.protein=='mTQ2') & (averaged_base_spectra_df.temp==row['temp'])].iloc[:1,6:157].T.iloc[:,0]
    ng_base_spec = averaged_base_spectra_df[(averaged_base_spectra_df.protein=='mNG') & (averaged_base_spectra_df.temp==row['temp'])].iloc[:1,6:157].T.iloc[:,0]
    y_raw = row[8:159].to_numpy()
    y_raw_normalized = y_raw / y_raw.sum()
    y_corr_normalized = ng_base_spec / (avg_idr_dirA_intensity_ratio * ng_base_spec.sum())
    y_corrected = y_raw_normalized - y_corr_normalized
    E_f, d_over_a = calculate_monomer_efret(y_corrected, tq_base_spec, ng_base_spec)
    E_f_results_df = E_f_results_df.append({'category':row['category'],'idr':row['idr'],'E_f':E_f}, ignore_index=True)

# plot graph
idrs = ['GS16', 'p53', 'Ash1', 'E1A']
idr_indices = ['1-GS16', '2-p53', '3-Ash1', '4-E1A']
categories = ['Original', 'Flipped']
category_indices = ['1-Original', '2-Flipped']
colors = ['grey', 'royalblue', 'gold', 'teal', 'grey', 'royalblue', 'gold', 'teal']
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10,10))
for i in range(len(categories)):
    for j in range(len(idrs)):
        curr_E_f_mean = E_f_results_df[(E_f_results_df.category==categories[i]) & (E_f_results_df.idr==idrs[j])].E_f.mean()
        curr_E_f_err = E_f_results_df[(E_f_results_df.category==categories[i]) & (E_f_results_df.idr==idrs[j])].E_f.std()
        graph_df = graph_df.append({'category':categories[i],'category_index':category_indices[i],'idr':idrs[j],'idr_index':idr_indices[j],
                                    'E_f_mean':curr_E_f_mean,'E_f_err':curr_E_f_err}, ignore_index=True)
# unstack data to get bar charts by idr
data = graph_df.set_index(['idr_index', 'category_index'])
# Will use errorbar for error bars over swarm plots.
data['E_f_mean'].unstack().plot(kind='bar', capsize=4, ax=ax, width=0.8)
bars = ax.patches
for i in range(len(bars)):
    bars[i].set_facecolor(colors[i])
    if i > 3:
        bars[i].set_alpha(0.5)
    bars[i].set_edgecolor('black')
    bars[i].set_linewidth(3)
    
ax.set_xlabel(None)
ax.set_xticks([0,1,2,3], idrs)
ax.legend(loc='upper center', fontsize=30, frameon=False, labels=['Original','Flipped'])
plt.xticks(rotation=0)
ax.set_ylabel('$E_f^{app}$', labelpad=0)
ax.set_ylim([0, 0.8])
ax.grid(visible=True)
    
# Swarm plots
for i in range(len(idrs)):
    for j in range(len(categories)):
        curr_E_fs = E_f_results_df[(E_f_results_df.idr==idrs[i]) & (E_f_results_df.category==categories[j])].E_f
        curr_E_f_mean = E_f_results_df[(E_f_results_df.idr==idrs[i]) & (E_f_results_df.category==categories[j])].E_f.mean()
        curr_E_f_err = E_f_results_df[(E_f_results_df.idr==idrs[i]) & (E_f_results_df.category==categories[j])].E_f.std()
        if j == 0:
            x_offset = -0.2
        else:
            x_offset = 0.2
        for E_f in curr_E_fs:
            ax.scatter(i + x_offset - 0.14 + 0.28*random.random(), E_f, color=colors[i], alpha=0.5-(0.25*j), s=200, lw=2, edgecolor='black', zorder=90)
        ax.errorbar(i + x_offset, curr_E_f_mean, yerr=curr_E_f_err, capsize=4, capthick=2, ls='none', color='black', elinewidth=3, zorder=100)
    
plt.show()
plt.savefig('Fig_5B.png', bbox_inches='tight')
# plt.savefig('Fig_5B.svg', bbox_inches='tight')

# graph_df[['category','idr','E_f_mean','E_f_err']].to_csv('structural-bias-flipped-idrs-average-fret.csv', index=False)
# E_f_results_df.to_csv('structural-bias-nsmb-fig-5b-flipped-idrs-fret.csv', index=False)

