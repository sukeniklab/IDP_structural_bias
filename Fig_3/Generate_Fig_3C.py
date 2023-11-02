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
mpl.rcParams['ytick.labelsize'] = 55
mpl.rcParams['xtick.labelsize'] = 55
mpl.rcParams['ytick.minor.size'] = 10
mpl.rcParams['ytick.minor.width'] = 7
mpl.rcParams['font.size'] = 55
mpl.rcParams['font.sans-serif']='Arial'
plt.rcParams['errorbar.capsize']=4

# Create and format the figure
ncols=1
nrows=1
wspace=0
fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharex=False, sharey=False, 
                        figsize=(ncols*11 + (wspace*(ncols-1)), nrows*10))
plt.subplots_adjust(wspace=wspace)
plt.margins(x=0,y=0)

###############
# Fig 3C
###############

csvs = ['structural-bias-nsmb-fig-3c-PUMA_WT_UV280_normalized.csv',
        'structural-bias-nsmb-fig-3c-PUMA_S1_UV280_normalized.csv',
        'structural-bias-nsmb-fig-3c-PUMA_S2_UV280_normalized.csv',
        'structural-bias-nsmb-fig-3c-PUMA_S3_UV280_normalized.csv']
labels = ['WT', 'S1', 'S2', 'S3']
colors = ['darkslateblue','blueviolet','violet','darkmagenta']
for i in range(len(csvs)):
    df = pd.read_csv(csvs[i])
    ax.plot(df.wavelength[:600], df['1uM_10mm'][:600], color=colors[i], label=labels[i], linestyle='-', linewidth=7)
ax.set_xlabel('wavelength (nm)')
ax.set_xlim([200,260])
ax.set_ylabel('MRE (10$^3$ deg cm$^2$ dmol$^{-1}$)', rotation=90, labelpad=0, fontsize=50, loc='top')
ax.legend(fontsize=36, frameon=False, borderpad=0.1, loc='center right') 
ax.grid(visible=True)

fig.tight_layout()
plt.savefig('structural_bias_nsmb_fig_3C.png', bbox_inches='tight')
plt.show()

