# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 20:56:55 2023

@author: Karina Guadalupe
"""
#%% library imports
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 
import matplotlib as mpl
from matplotlib.colors import to_rgb
from matplotlib.collections import PolyCollection
from matplotlib.legend_handler import HandlerTuple
import matplotlib.lines as mlines

def jitter(arr):
    stdev = .011 * (max(arr) - min(arr))
    return arr + np.random.randn(len(arr)) * stdev
#%% general parameters for all figures
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
TableS3=TableS3.query('Efret_before > Efret_before.quantile(.05)')
TableS3['FPorder']= TableS3['FPorder'].astype('category')
#%% Fig 5C
order=['GS16','p53','Ash1','E1A']
colors=['grey','royalblue','gold','teal']
handles = []
fig,ax=plt.subplots(figsize=(10,10))
ax.grid()
ax = sns.violinplot(data=TableS3, x=TableS3["construct"], y=TableS3['Efret_before'], hue=TableS3["FPorder"], hue_order=['O', 'F'], order=order)

lines_25_75=[0,2,4,6,8,10,12,14]
median50_box=[1,3,5,7,9,11,13,15]   
for line in lines_25_75:
    ax.lines[line].set_linewidth(3)
for box in median50_box:
    ax.lines[box].set_color('black')
    ax.lines[box].set_linewidth(13)
    points=ax.collections[box]
    size=points.get_sizes().item()
    new_size=[size*13]
    points.set_sizes(new_size)
    
for ind, violin in enumerate(ax.findobj(PolyCollection)):
    rgb = to_rgb(colors[ind // 2])
    if ind % 2 != 0:
        rgb = 0.5 + 0.5 * np.array(rgb)  # make whiter
    violin.set_facecolor(rgb)
    handles.append(plt.Rectangle((0, 0), 0, 0, facecolor=rgb, edgecolor='black'))

ax.legend(handles=[tuple(handles[1::2]), tuple(handles[::2])], labels=TableS3["FPorder"].cat.categories.to_list(),
            title="FP Order", handlelength=4, handler_map={tuple: HandlerTuple(ndivide=None, pad=0)},loc='right',bbox_to_anchor=(1.75,.51))
plt.ylim(0.0,0.8)
plt.xticks(rotation=0,ha='center')
plt.ylabel('$E_f^{cell}$',size=60)
plt.xlabel(' ')
fig.savefig('Fig.5C.svg', format="svg",bbox_inches='tight', dpi=1200)
#%% Fig 5D
order=['p53','Ash1','E1A']
colors=['royalblue','gold','teal']
#perturbed
n=len(order)
fig,ax= plt.subplots(1,3,sharex=True,sharey=True,figsize=(22,10))
fig.subplots_adjust(hspace=0.0, wspace=0.3)
for orderIdx in range(len(order)):
    idr=order[orderIdx]
    sliced = TableS3[(TableS3.construct==idr)&(abs(TableS3['ch7']-TableS3['ch3'])<2000)]
    sliced=sliced[sliced['condition']!=300]
    col=int(orderIdx)
    ax[col].grid()
    sns.violinplot(x='condition',y='deltaEfret',data=sliced,hue='FPorder',hue_order=['O','F'],
                     ax=ax[col],inner='box',color=colors[orderIdx])
    lines_25_75=[0,2,4,6]
    median50_box=[1,3,5,7]   
    for line in lines_25_75:
        ax[col].lines[line].set_linewidth(3)
    for box in median50_box:
        ax[col].lines[box].set_color('black')
        ax[col].lines[box].set_linewidth(13)
        points=ax[col].collections[box]
        size=points.get_sizes().item()
        new_size=[size*13]
        points.set_sizes(new_size)
    for ind, violin in enumerate(ax[col].findobj(PolyCollection)):
        rgb = to_rgb(colors[orderIdx//1])
        if ind % 2 != 0:
            rgb = 0.5 + 0.5 * np.array(rgb) # make whiter
        else: rgb=rgb
        violin.set_facecolor(rgb)
        # Add legend with custom labels
        custom_legend_labels = [mlines.Line2D([0], [0], marker='s', color='w', label='Original', markerfacecolor=colors[orderIdx], markersize=30),
                                mlines.Line2D([0], [0], marker='s', color='w', label='Flipped', markerfacecolor=(0.5 + 0.5 * np.array(to_rgb(colors[orderIdx]))), markersize=30)]     
        ax[col].legend(handles=custom_legend_labels, loc="upper left", fontsize=30, frameon=False)
    ax[col].axhline(y = 0, color = 'grey', linestyle = '-')
    ax[col].text(0.5, 0.045, idr, horizontalalignment='center',fontsize=50, verticalalignment='center', transform=ax[col].transAxes)
    ax[col].set_ylabel(' ')
    ax[col].set_xlabel(' ')
plt.ylim(-0.05,0.1)
fig.text(0.5, -0.06, 'osmotic challenge (mOsm)', ha='center',size=60)
fig.text(0.007, 0.5, '\u0394 $E_f^{cell}$', va='center', rotation='vertical',size=60)
fig.savefig('Fig.5D.svg', format="svg",bbox_inches='tight', dpi=1200)