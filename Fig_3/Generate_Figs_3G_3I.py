# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 16:00:42 2023

@author: Karina Guadalupe
"""
#%% library imports
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 
import matplotlib as mpl
from matplotlib.colors import to_rgb
from matplotlib.legend_handler import HandlerTuple
from statannot import add_stat_annotation
from matplotlib.patches import PathPatch
#%% general figure formatting
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
#%% data import
TableS3=pd.read_csv('Table_S3.csv')
colors = pd.read_csv('hidden_structure_color_scheme.csv')#,usecols=['protein','color','N'])
colors = colors.rename(columns={'protein': 'construct'})
#data cleanup - explained in the methods section
# cleanup
TableS3 = TableS3[TableS3['D/A_before']<6]
TableS3 = TableS3[TableS3['Area']<650]
TableS3 = TableS3[TableS3['ch1']>3000]
TableS3 = TableS3[TableS3['ch3']<10000]
##selecting original constructs (not the ones with the flipped FP pair)
TableS3=TableS3[TableS3['FPorder']=='O']
#%% defining variables 
prots = ['GS0','GS16','GS32','GS48','GS64','Ash1', 'E1A', 'FUS','p53','PUMA WT', 'PUMA S1', 'PUMA S2', 'PUMA S3']
GSs = ['GS0','GS16','GS32','GS48','GS64']
others = ['PUMA WT','Ash1', 'E1A', 'FUS','p53']
PUMAs = ['PUMA WT', 'PUMA S1', 'PUMA S2', 'PUMA S3']
osms = [100,300,750]
#%% fits 
allMeans = pd.DataFrame()
means=np.array([])
errs=np.array([])
## basal Efret fits using GS linkers
for protIdx,prot in enumerate(GSs):
    sliced = TableS3[TableS3.construct==prot]
    color = colors[colors['construct']==prot]['color']
    N_res = int(prot[2:])*2
    bot, quartile1, medians, quartile3, top = np.percentile(sliced['Efret_before'], [5, 25, 50, 75, 95])
    allMeans = allMeans.append({'color':color.values[0],'prot':prot,'q1':quartile1,'median':medians,'q3':quartile3},ignore_index=True)
    means = np.append(means,medians)
    errs = np.append(errs,quartile3-quartile1)
Ef_GS_x=np.array([int(x[2:])*2 for x in iter(GSs)])
fit,cov = np.polyfit(Ef_GS_x,means,1,cov=True,w=1/errs)
fit_err = np.sqrt(np.diag(cov))
Ef_GS_x=np.append(Ef_GS_x,200)
Ef_GS_y=Ef_GS_x*fit[0]+fit[1]
Ef_GS_y_top = Ef_GS_x*(fit[0]+fit_err[0])+fit[1]+fit_err[1]
Ef_GS_y_bot = Ef_GS_x*(fit[0]-fit_err[0])+fit[1]-fit_err[1]

## deltaFRET fits using deltaEF of linkers
for osmIdx,osm in enumerate(osms):
    meds = np.array([])
    errs = np.array([])
    x = np.array([])
    for protIdx,prot in enumerate(GSs):
        N_res = int(prot[2:])*2
        color = colors[colors['construct']==prot]['color']
        sliced = TableS3[(TableS3.construct==prot)&(abs(TableS3['ch7']-TableS3['ch3'])<2000)]
        sliced = sliced[sliced['condition']==osm]
        bot, quartile1, medians, quartile3, top = np.percentile(sliced['deltaEfret'], [5, 25, 50, 75, 95])
        meds = np.append(meds,medians)
        errs = np.append(errs,(quartile3-quartile1))
        x = np.append(x,N_res)
    fit_deltaEfret,cov= np.polyfit(x,meds,1,cov=True,w=None)
    fit_deltaEfret_err = np.sqrt(np.diag(cov))
    X = np.linspace(0,200)
    Y = X*fit_deltaEfret[0]+fit_deltaEfret[1]
    Y_top = X*(fit_deltaEfret[0]+fit_deltaEfret_err[0])+fit_deltaEfret[1]+fit_deltaEfret_err[1]
    Y_bot = X*(fit_deltaEfret[0]-fit_deltaEfret_err[0])+fit_deltaEfret[1]-fit_deltaEfret_err[1]
    
#%% Fig 3G
fig,ax = plt.subplots(1,1, figsize=[10,10], sharex=True, sharey=True)
ax.grid()
for protIdx,prot in enumerate(PUMAs):
    sliced = TableS3[TableS3.construct==prot]
    color = colors[colors['construct']==prot]['color']
    N_res = colors[colors['construct']==prot]['N']
    E_f_IV = 1/(1+(colors[colors['construct']==prot]['d_over_a_normalized_to_gs_equiv']))
    meanGS = (N_res*fit[0]+fit[1]).values
    meanGS_top = (N_res*(fit[0]+fit_err[0])+fit[1]+fit_err[1]).values
    meanGS_bot = (N_res*(fit[0]-fit_err[0])+fit[1]-fit_err[1]).values
    parts = ax.violinplot(sliced['Efret_before'],positions=[protIdx],
                          widths=0.5,showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor(color)
        pc.set_edgecolor('black')
        pc.set_lw(3)
        pc.set_alpha(1)
    ax.scatter(protIdx,(sliced['Efret_before']).median(),marker="_",
               c='w',s=200,zorder=3,linewidth=5,edgecolor='k')
    means = np.append(means,sliced['Efret_before'].mean())
    ax.plot([-1,6],[meanGS,meanGS],'--',c='cadetblue',lw=5,zorder=0)
    ax.fill_between([-1,6],meanGS_top,meanGS_bot,color='cadetblue',alpha=0.1,zorder=0)
    bot, quartile1, medians, quartile3, top = np.percentile(sliced['Efret_before'], [5, 25, 50, 75, 95])
    allMeans = allMeans.append({'color':color.values[0],'prot':prot,'q1':quartile1,'median':medians,'q3':quartile3},ignore_index=True)
    ax.vlines(protIdx, quartile1, quartile3, color='k', linestyle='-', lw=10)
    ax.vlines(protIdx, bot, top, color='k', linestyle='-', lw=3)
ax.set_xlim(-0.5,3.5)
ax.set_xticks(range(len(PUMAs)))
ax.set_xticklabels(['WT','S1','S2','S3'],fontsize=50)
ax.set_ylim(0.0,0.8)
ax.set_yticks([0,0.2,0.4,0.6,0.8])
ax.set_ylabel('$E_f^{cell}$',fontsize=60)
fig.savefig('Fig.3G.svg', format="svg",bbox_inches='tight', dpi=1200)

#%% Figure 3I
fig,ax = plt.subplots(1,4, figsize=[26,10], sharex=True, sharey=True)
for protIdx,prot in enumerate(PUMAs):
    color = colors[colors['construct']==prot]['color']
    for osmIdx,osm in enumerate(osms):
        N_res = colors[colors['construct']==prot]['N']
        if osm==750:
            deltaEfret_GS = N_res*fit_deltaEfret[0]+fit_deltaEfret[1]
            deltaEfret_GS_err = N_res*fit_deltaEfret_err[0]+fit_deltaEfret_err[1]
        else:   
            deltaEfret_GS=0
            deltaEfret_GS_err=0.002
        sliced = TableS3[(TableS3.construct==prot)&(abs(TableS3['ch7']-TableS3['ch3'])<2000)]
        sliced = sliced[sliced['condition']==osm]
        parts = ax[protIdx].violinplot(sliced['deltaEfret'],positions=[osmIdx+0.2],
                          widths=0.5,showextrema=False)
        for pc in parts['bodies']:
            pc.set_facecolor(color)
            pc.set_edgecolor('black')
            pc.set_lw(3)
            pc.set_alpha(1)
        ax[protIdx].scatter(osmIdx+0.2,(sliced['deltaEfret']).median(),marker='_',
                            c='w',s=100,zorder=3,linewidth=10,edgecolor='k')
        ax[protIdx].grid()
        bot, quartile1, medians, quartile3, top = np.percentile(sliced['deltaEfret'], [5, 25, 50, 75, 95])
        ax[protIdx].vlines(osmIdx+0.2, quartile1, quartile3, color='k', linestyle='-', lw=10)
        ax[protIdx].vlines(osmIdx+0.2, bot, top, color='k', linestyle='-', lw=3)
        ax[protIdx].scatter(osmIdx-0.2,deltaEfret_GS,s=1000,marker='s',c=color,alpha=0.3,linewidth=5,edgecolor='k',zorder=3)
        ax[protIdx].errorbar(osmIdx-0.2,deltaEfret_GS,yerr=deltaEfret_GS_err,marker=None,color='k',linewidth=5,zorder=3)
        ax[protIdx].text(0.03,0.985,prot,ha='left',va='top',backgroundcolor='w',transform=ax[protIdx].transAxes)

ax[0].set_xlim([-0.5,2.5])
ax[0].set_ylabel(r'$\Delta E_f^{cell}$',fontsize=60)
ax[0].set_ylim(-0.05,0.1)
ax[0].set_yticks([-0.05,0,0.05,0.1])
ax[0].set_xticks(range(len(osms)))
ax[0].set_xticklabels([100,300,750])
fig.text(0.5,-0.02,'osmotic challenge (mOsm)',va='top',ha='center',fontsize=60)
fig.savefig('Fig.3I.svg', format="svg",bbox_inches='tight', dpi=1200)
