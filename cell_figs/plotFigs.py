# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 21:33:37 2022

@author: ssukenik
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl


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

#%% imports

allcells = pd.read_csv('../Table_S2.csv',index_col=0)
cells = allcells.copy()
colors = pd.read_csv('../hidden_structure_color_scheme.csv')

prots = ['GS0','GS16','GS32','GS48','GS64','Ash1', 'E1A', 'FUS','p53','PUMA WT', 'PUMA S1', 'PUMA S2', 'PUMA S3']
GSs = ['GS0','GS16','GS32','GS48','GS64']
others = ['PUMA WT','Ash1', 'E1A', 'FUS','p53']
PUMAs = ['PUMA WT', 'PUMA S1', 'PUMA S2', 'PUMA S3']
osms = [100,300,750]

allMeans = pd.DataFrame()

# cleanup
cells = cells[cells['D/A_before']<6]
cells = cells[cells['Area']<650]
cells = cells[cells['ch1']>3000]
cells = cells[cells['ch3']<10000]


#%% Figure S7
fig,ax = plt.subplots(4,4,figsize=[30,30],sharex=True,sharey=True)
for protIdx,prot in enumerate(prots):
    sliced = cells[cells.construct==prot]
    color = colors[colors['protein']==prot]['color']
    col = protIdx%int(np.ceil(np.sqrt(len(prots))))
    row = int(np.floor(protIdx/4))
    ax[row,col].hist(sliced['Area'],50,range=(0,700),density=True,color=color)
    ax[row,col].text(0.98,0.85,prot,transform=ax[row,col].transAxes,ha='right',
                     fontsize=60)
    ax[row,col].tick_params(axis='both', which='major', labelsize=80)
    ax[row,col].grid(b=True)
fig.text(0.5,0.05,'cell area (pixels)',ha='center',va='top',fontsize=120)
fig.text(0,0.5,'density',rotation=90,va='center',ha='left',fontsize=120)
fig.savefig('Fig.S7_area.svg')

fig,ax = plt.subplots(4,4,figsize=[30,30],sharex=True,sharey=True)
for protIdx,prot in enumerate(prots):
    sliced = cells[cells.construct==prot]
    color = colors[colors['protein']==prot]['color']
    col = protIdx%int(np.ceil(np.sqrt(len(prots))))
    row = int(np.floor(protIdx/4))
    ax[row,col].hist(sliced['Round'],50,range=(0,1),density=True,color=color)
    ax[row,col].text(0.98,0.85,prot,transform=ax[row,col].transAxes,ha='right',
                     fontsize=60)
    ax[row,col].tick_params(axis='both', which='major', labelsize=80)
    ax[row,col].grid(b=True)
fig.text(0.5,0.05,'sphericity',ha='center',va='top',fontsize=120)
fig.text(0.02,0.5,'density',rotation=90,va='center',ha='left',fontsize=120)
fig.savefig('Fig.S7_sphericity.svg')


fig,ax = plt.subplots(4,4,figsize=[30,30],sharex=True,sharey=True)
for protIdx,prot in enumerate(prots):
    sliced = allcells[allcells.construct==prot]
    color = colors[colors['protein']==prot]['color']
    col = protIdx%int(np.ceil(np.sqrt(len(prots))))
    row = int(np.floor(protIdx/4))
    ax[row,col].hist(sliced['ch3']/1e3,50,range=(0,20),density=True,color=color)
    ax[row,col].text(0.95,0.85,prot,transform=ax[row,col].transAxes,ha='right',
                     fontsize=60,backgroundcolor='w')
    ax[row,col].plot([1e1,1e1],[0,1],'--',c='k',lw=3)
    ax[row,col].tick_params(axis='both', which='major', labelsize=80)
    ax[row,col].grid(b=True)
ax[0,0].set_ylim(0,3e-1)
fig.text(0.5,0.05,'$10^3$ direct acceptor emission',ha='center',va='top',fontsize=120)
fig.text(0.02,0.5,'density',rotation=90,va='center',ha='right',fontsize=120)
fig.savefig('Fig.S7_ch3.svg')


#%% Fig. 2G

fig,ax = plt.subplots(1,1, figsize=[10,10], sharex=True, sharey=True)
means=np.array([])
errs=np.array([])
ax.grid(b=True)
for protIdx,prot in enumerate(GSs):
    sliced = cells[cells.construct==prot]
    color = colors[colors['protein']==prot]['color']
    N_res = int(prot[2:])*2
    E_f_IV = colors[colors['protein']==prot]['E_f']
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
    # ax.scatter(N_res,E_f_IV,alpha=1,marker='s',
    #            c='w',s=300,zorder=3,linewidth=5,edgecolor='k')
    means = np.append(means,medians)
    errs = np.append(errs,quartile3-quartile1)
    # ax.text(N_res,0.89,str(len(sliced)),ha='center',fontsize=25)
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
fig.savefig('Fig.2G.svg', bbox_inches='tight')

#%% Fig. 2H

fig,ax = plt.subplots(1,len(GSs), figsize=[26,10], sharex=True, sharey=True)
for protIdx,prot in enumerate(GSs):
    for osmIdx,osm in enumerate(osms):
        color = colors[colors['protein']==prot]['color']
        sliced = cells[(cells.construct==prot)&(abs(cells['ch7']-cells['ch3'])<2000)]
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
        ax[protIdx].grid(b=True)
        bot, quartile1, medians, quartile3, top = np.percentile(sliced['deltaEfret'], [5, 25, 50, 75, 95])
        ax[protIdx].vlines(osmIdx, quartile1, quartile3, color='r', linestyle='-', lw=10)
        ax[protIdx].vlines(osmIdx, bot, top, color='r', linestyle='-', lw=3)
    ax[protIdx].text(0.03,0.985,prot,ha='left',va='top',backgroundcolor='w',transform=ax[protIdx].transAxes)
    
ax[0].set_ylabel(r'$\Delta E_f^{cell}$',fontsize=60)
ax[0].set_ylim(-0.05,0.2)
ax[0].set_xticks(range(len(osms)))
ax[0].set_xticklabels([100,300,750])
fig.text(0.5,-0.02,'osmotic challenge (mOsm)',va='top',ha='center',fontsize=60)
fig.savefig('Fig.2H.svg', bbox_inches='tight')
#%% Figure S8

fig,ax=plt.subplots(1,3,figsize=[20,10],sharex=True,sharey=True)
for osmIdx,osm in enumerate(osms):
    meds = np.array([])
    errs = np.array([])
    x = np.array([])
    for protIdx,prot in enumerate(GSs):
        N_res = int(prot[2:])*2
        color = colors[colors['protein']==prot]['color']
        sliced = cells[(cells.construct==prot)&(abs(cells['ch7']-cells['ch3'])<2000)]
        sliced = sliced[sliced['condition']==osm]
        parts = ax[osmIdx].violinplot(sliced['deltaEfret'],positions=[N_res],
                          widths=20,showextrema=False)
        for pc in parts['bodies']:
            pc.set_facecolor(color)
            pc.set_edgecolor('black')
            pc.set_lw(3)
            pc.set_alpha(1)
        ax[osmIdx].scatter(N_res,(sliced['deltaEfret']).median(),marker='_',
                            c='w',alpha=1,s=200,zorder=3,linewidth=5,edgecolor='k')
        ax[osmIdx].grid(b=True)
        bot, quartile1, medians, quartile3, top = np.percentile(sliced['deltaEfret'], [5, 25, 50, 75, 95])
        ax[osmIdx].vlines(N_res, quartile1, quartile3, color='r', linestyle='-', lw=10)
        ax[osmIdx].vlines(N_res, bot, top, color='r', linestyle='-', lw=3)
        meds = np.append(meds,medians)
        errs = np.append(errs,(quartile3-quartile1))
        x = np.append(x,N_res)
    fit_deltaEfret,cov= np.polyfit(x,meds,1,cov=True,w=None)
    fit_deltaEfret_err = np.sqrt(np.diag(cov))
    X = np.linspace(0,200)
    Y = X*fit_deltaEfret[0]+fit_deltaEfret[1]
    Y_top = X*(fit_deltaEfret[0]+fit_deltaEfret_err[0])+fit_deltaEfret[1]+fit_deltaEfret_err[1]
    Y_bot = X*(fit_deltaEfret[0]-fit_deltaEfret_err[0])+fit_deltaEfret[1]-fit_deltaEfret_err[1]
    ax[osmIdx].plot(X,X*fit_deltaEfret[0]+fit_deltaEfret[1],'--',lw=5,c='cadetblue')
    ax[osmIdx].fill_between(X,Y_bot,Y_top,color='cadetblue',alpha=0.2)
    ax[osmIdx].text(0.03,0.035,str(osm)+" mOsm",ha='left',backgroundcolor='w',transform=ax[osmIdx].transAxes)

ax[0].set_ylabel(r'$\Delta E_f^{cell}$',fontsize=60)
ax[0].set_ylim(-0.05,0.15)
ax[0].set_xlim(-20,160)
ax[0].set_xticks([0,50,100,150])
fig.text(0.5,-0.02,'$N_{residues}$',va='top',ha='center',fontsize=60)
fig.savefig('Fig.S8.svg', bbox_inches='tight')
#%% Figure 4D

fig,ax = plt.subplots(1,1, figsize=[10,10], sharex=True, sharey=True)
ax.grid(b=True,zorder=3)
for protIdx,prot in enumerate(others):
    sliced = cells[cells.construct==prot]
    color = colors[colors['protein']==prot]['color']
    N_res = colors[colors['protein']==prot]['N']
    E_f_IV = colors[colors['protein']==prot]['E_f']
    meanGS = (N_res*fit[0]+fit[1]).values
    parts = ax.violinplot(sliced['Efret_before'],positions=N_res,
                          widths=15,showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor(color)
        pc.set_edgecolor('black')
        pc.set_lw(3)
        pc.set_alpha(1)
    bot, quartile1, medians, quartile3, top = np.percentile(sliced['Efret_before'], [5, 25, 50, 75, 95])
    allMeans = allMeans.append({'color':color.values[0],'prot':prot,'q1':quartile1,'median':medians,'q3':quartile3},ignore_index=True)
    ax.vlines(N_res, quartile1, quartile3, color='k', linestyle='-', lw=10)
    ax.vlines(N_res, bot, top, color='k', linestyle='-', lw=3)
    ax.scatter(N_res,medians,c='w',marker='s',s=80,zorder=3,edgecolor='k',lw=3)
ax.plot(Ef_GS_x,Ef_GS_y,'--',lw=5,c='cadetblue',zorder=0)
ax.fill_between(Ef_GS_x,Ef_GS_y_bot,Ef_GS_y_top,color='cadetblue',alpha=0.1,zorder=0)
ax.set_ylim(0,0.8)
ax.set_xlim(25,175)
ax.set_xticks([50,100,150])
ax.set_ylabel('$E_f^{cell}$',fontsize=60)
ax.set_xlabel('$N_{residues}$',fontsize=60)
fig.savefig('Fig.4D.svg', bbox_inches='tight')

#%% Figure 4F

fig,ax = plt.subplots(1,5, figsize=[30,10], sharex=True, sharey=True)
for protIdx,prot in enumerate(others):
    color = colors[colors['protein']==prot]['color']
    for osmIdx,osm in enumerate(osms):
        N_res = colors[colors['protein']==prot]['N']
        if osm==750:
            deltaEfret_GS = N_res*fit_deltaEfret[0]+fit_deltaEfret[1]
            deltaEfret_GS_err = N_res*fit_deltaEfret_err[0]+fit_deltaEfret_err[1]
        else:   
            deltaEfret_GS=0
            deltaEfret_GS_err=0.002
        sliced = cells[(cells.construct==prot)&(abs(cells['ch7']-cells['ch3'])<2000)]
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
        ax[protIdx].grid(b=True)
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
fig.savefig('Fig.4F.svg', bbox_inches='tight')
#%% Figure 3G

fig,ax = plt.subplots(1,1, figsize=[10,10], sharex=True, sharey=True)
ax.grid(b=True,zorder=3)
for protIdx,prot in enumerate(PUMAs):
    sliced = cells[cells.construct==prot]
    color = colors[colors['protein']==prot]['color']
    N_res = colors[colors['protein']==prot]['N']
    E_f_IV = 1/(1+(colors[colors['protein']==prot]['d_over_a_normalized_to_gs_equiv']))
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
fig.savefig('Fig.3G.svg', bbox_inches='tight')

#%% Figure 3I

fig,ax = plt.subplots(1,4, figsize=[26,10], sharex=True, sharey=True)
for protIdx,prot in enumerate(PUMAs):
    color = colors[colors['protein']==prot]['color']
    for osmIdx,osm in enumerate(osms):
        N_res = colors[colors['protein']==prot]['N']
        if osm==750:
            deltaEfret_GS = N_res*fit_deltaEfret[0]+fit_deltaEfret[1]
            deltaEfret_GS_err = N_res*fit_deltaEfret_err[0]+fit_deltaEfret_err[1]
        else:   
            deltaEfret_GS=0
            deltaEfret_GS_err=0.002
        sliced = cells[(cells.construct==prot)&(abs(cells['ch7']-cells['ch3'])<2000)]
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
        ax[protIdx].grid(b=True)
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
fig.savefig('Fig.3I.svg', bbox_inches='tight')