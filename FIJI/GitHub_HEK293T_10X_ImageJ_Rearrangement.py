# -*- coding: utf-8 -*-
"""
Created on Wed May  4 23:05:40 2022

@author: guada
"""
#%% library imports
import pandas as pd
import os
from pathlib import Path
#%% getting current working directory
cwd = os.getcwd()
rootdir=cwd.replace("\\", "/")
# rootdir = Path.cwd()
#%% definining function to get list of directories 
def listdirs(rootdir):
    alist=[]
    for it in os.scandir(rootdir):
        if it.is_dir():
            string=(str(it.path)).split('\\')[-1]
            alist+=[string]
    return alist
#%% defining function to rearrange each output from FIJI ImageJ
def rearrange(threshold):
    subdirectorylist=listdirs(rootdir+"/images")
    for T in range(len(subdirectorylist)):
        # experimentname=subdirectorylist[T].split('\\')[1]
        experimentname=subdirectorylist[T]
        print('This is the first directory: '+str(T)+' and the folder name is: '+subdirectorylist[T])
        # importing the results and conditions info
        path2results=rootdir+"/images/"
        path_to_conditions=rootdir+"/layouts/"
        # path2results = os.path.join(rootdir, "images")
        # path_to_conditions = os.path.join(rootdir, "layouts")
        #insert channel numbers, split the file name, and add column for cell type
        # resultsfoldername=threshold+"_Results"
        # csvname=experimentname+"_"+threshold+"_Results.csv"
        # csvpath=os.path.join(path2results, experimentname, resultsfoldername, csvname)
        # df=pd.read_csv(csvpath)
        df=pd.read_csv(path2results+experimentname+"/"+threshold+"_Results/"+experimentname+"_Results.csv")
        df=df.drop(labels=' ', axis=1)
        #inserting channel col in the beginning
        df.insert(0,'channel',(['ch1','ch2','ch3','ch4','ch5','ch6','ch7','ch8']*int(((len(df)))/8)))
        df[['date','mag','cellline','trial','FPorder','notes','well']]=df['Label'].str.split('_',expand=True)
        # rearranging to have channels as columns
        df=df.pivot_table(index=['Label','date','mag','cellline','trial','FPorder','notes','well','channel'],columns=[],dropna=True)
        df=pd.DataFrame.copy(df).reset_index()
        df=pd.wide_to_long(df, stubnames=['Area','Mean','StdDev','Circ.','AR','Round','Solidity'],
                          i=['Label','well','date','mag','cellline','trial','FPorder','channel'], 
                          j='cell').sort_values(['Label','well','date','mag','cellline','trial','channel']).dropna(how='all')
        df=df.reset_index()
        df=df.pivot_table(index=['Label','well','date','mag','cellline','trial','FPorder','cell','Area','Circ.',
                                  'AR','Round','Solidity'],columns='channel',dropna=True)
        #flattening multilevel index to combine. Ex: ch1 and ch2 with Std is Std_ch1 and Std_ch2 
        df.columns = df.columns.get_level_values(0) + '_' +  df.columns.get_level_values(1)
        #renaming Mean_ch cols to just ch
        df.rename(columns={'Mean_ch1': 'ch1', 'Mean_ch2': 'ch2_uncorrected', 'Mean_ch3': 'ch3', 'Mean_ch4': 'ch4','Mean_ch5': 'ch5', 'Mean_ch6': 'ch6_uncorrected', 'Mean_ch7': 'ch7', 'Mean_ch8': 'ch8'}, inplace=True)
        #correcting for bleedthrough and crossexcitation
        livecellbleedthrough=0.53
        crossexcitation=0.19
        df['ch2']=df['ch2_uncorrected']-(df['ch1']*livecellbleedthrough)-(df['ch3']*crossexcitation)
        df['D/A_before']=df['ch1']/df['ch2']
        df['ch6']=df['ch6_uncorrected']-(df['ch5']*livecellbleedthrough)-(df['ch7']*crossexcitation)
        df['D/A_after']=df['ch5']/df['ch6']
        df['Efret_before']=df['ch3']/(df['ch1']+df['ch2'])
        df['Efret_after']=df['ch6']/(df['ch5']+df['ch6'])
        df['deltaEfret']=df['Efret_after']-df['Efret_before']
        #this script is for hek293 images taken at 10X
        df['location']='whole cell'
        #resetting index to get other info as columns
        df=df.reset_index()
        #dropping rows that contain zeros
        df=df[df['Area'] != 0]
        #merging to construct and condition
        conditions=pd.read_csv(path_to_conditions+"/"+experimentname+"_conditions.csv")
        df=pd.merge(df,conditions,on=["well"],how='outer').dropna()
        #selecting the final columns for the output dataframe
        df=df[['dateImaged','cellline','location','mag','Area','Round','passage','construct','condition','well','trial',
                      'ch1','ch2_uncorrected','ch2', 'ch3','ch5', 'ch6_uncorrected','ch6', 'ch7',
                      'D/A_before','Efret_before','Efret_after','deltaEfret','FPorder']]
        if T==0:
            #creating a new csv file and adding the data
            df.to_csv(threshold+'_HEK293T_10X_Master_DataSet.csv', index=False, header = True)
        else:
            #appending data to already generate csv
            df.to_csv(threshold+'_HEK293T_10X_Master_DataSet.csv',mode='a', index=False, header = False)
#%% running the function
rearrange('setValues')
# rearrange('Triangle')
# rearrange('MinError')