# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 15:58:05 2023

@author: Karina Guadalupe
"""

#%% library imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import matplotlib as mpl
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
colors = pd.read_csv('hidden_structure_color_scheme.csv',usecols=['protein','color','N'])
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
#%% defining functions
allMeans = pd.DataFrame()
fig,ax = plt.subplots(1,1, figsize=[10,10], sharex=True, sharey=True)
means=np.array([])
errs=np.array([])
ax.grid()
for protIdx,prot in enumerate(GSs):
    sliced = TableS3[TableS3.construct==prot]
    color = colors[colors['construct']==prot]['color']
    N_res = int(prot[2:])*2
    parts = ax.violinplot(sliced['Efret_before'],positions=[N_res],
                          widths=20,showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor(color)
        pc.set_edgecolor('black')
        pc.set_lw(3)
        pc.set_alpha(1)
    bot, quartile1, medians, quartile3, top = np.percentile(sliced['Efret_before'], [5, 25, 50, 75, 95])
    allMeans = allMeans.append({'color':color.values[0],'prot':prot,'q1':quartile1,'median':medians,'q3':quartile3},ignore_index=True)
    ax.vlines(N_res, quartile1, quartile3, color='r', linestyle='-', lw=10)
    ax.vlines(N_res, bot, top, color='r', linestyle='-', lw=3)
    ax.scatter(N_res,medians,alpha=1,marker='s',
               c='w',s=80,zorder=3,linewidth=2,edgecolor='k')
    means = np.append(means,medians)
    errs = np.append(errs,quartile3-quartile1)
Ef_GS_x=np.array([int(x[2:])*2 for x in iter(GSs)])
fit,cov = np.polyfit(Ef_GS_x,means,1,cov=True,w=1/errs)
fit_err = np.sqrt(np.diag(cov))
Ef_GS_x=np.append(Ef_GS_x,200)
Ef_GS_y=Ef_GS_x*fit[0]+fit[1]
Ef_GS_y_top = Ef_GS_x*(fit[0]+fit_err[0])+fit[1]+fit_err[1]
Ef_GS_y_bot = Ef_GS_x*(fit[0]-fit_err[0])+fit[1]-fit_err[1]
ax.plot(Ef_GS_x,Ef_GS_y,'--',c='cadetblue',zorder=3,lw=5)
ax.fill_between(Ef_GS_x,Ef_GS_y_bot,Ef_GS_y_top,color='cadetblue',alpha=0.2,zorder=3)
ax.set_xlabel('$N_{residues}$',fontsize=60)
ax.set_ylabel('$E_f^{cell}$',fontsize=60)
ax.set_ylim(0,1)
ax.set_xlim(-10,170)
ax.set_xticks([0,50,100,150])
plt.savefig('Fig.2G.svg', format="svg",bbox_inches='tight', dpi=1200)
#%% Fig 2H
fig,ax = plt.subplots(1,len(GSs), figsize=[26,10], sharex=True, sharey=True)
for protIdx,prot in enumerate(GSs):
    for osmIdx,osm in enumerate(osms):
        color = colors[colors['construct']==prot]['color']
        sliced = TableS3[(TableS3.construct==prot)&(abs(TableS3['ch7']-TableS3['ch3'])<2000)]
        sliced = sliced[sliced['condition']==osm]
        parts = ax[protIdx].violinplot(sliced['deltaEfret'],positions=[osmIdx],
                          widths=0.5,showextrema=False)
        for pc in parts['bodies']:
            pc.set_facecolor(color)
            pc.set_edgecolor('black')
            pc.set_lw(3)
            pc.set_alpha(1)
        ax[protIdx].scatter(osmIdx,(sliced['deltaEfret']).median(),marker='_',
                            c='w',alpha=1,s=200,zorder=3,linewidth=5,edgecolor='k')
        ax[protIdx].grid()
        bot, quartile1, medians, quartile3, top = np.percentile(sliced['deltaEfret'], [5, 25, 50, 75, 95])
        ax[protIdx].vlines(osmIdx, quartile1, quartile3, color='r', linestyle='-', lw=10)
        ax[protIdx].vlines(osmIdx, bot, top, color='r', linestyle='-', lw=3)
    ax[protIdx].text(0.03,0.985,prot,ha='left',va='top',backgroundcolor='w',transform=ax[protIdx].transAxes)
    
ax[0].set_ylabel(r'$\Delta E_f^{cell}$',fontsize=60)
ax[0].set_ylim(-0.05,0.2)
ax[0].set_xticks(range(len(osms)))
ax[0].set_xticklabels([100,300,750])
fig.text(0.5,-0.02,'osmotic challenge (mOsm)',va='top',ha='center',fontsize=60)
plt.savefig('Fig.2H.svg', format="svg",bbox_inches='tight', dpi=1200)