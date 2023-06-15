########################################################################################################
# hidden_structure_data_processing.py
#
# Scripts to process FRET data.
#
# Author: David Moses (dmoses5@ucmerced.edu)
# 
# Supporting material for "Hidden structure in disordered proteins is adaptive to intracellular changes"
# 
# https://www.biorxiv.org/content/10.1101/2021.11.24.469609v1
########################################################################################################

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression

##############################################################################
# Import data from csv files
# Preprocess raw data for mNeonGreen and mTurquoise2 (FRET donor and acceptor)
##############################################################################
def preprocess_tq_ng_data():
    # Preprocess mTurquoise2 base spectra
    TQ_data_1=pd.read_csv("TQfree-1-repeat-1.csv",sep=",",skiprows=1).join(pd.read_csv('TQfree-2-repeat-1.csv',
                       skiprows=1).iloc[:,3:],rsuffix='2').join(pd.read_csv('TQfree-3-repeat-1.csv',
                       skiprows=1).iloc[:,3:],rsuffix='3')
    TQ_data_2=pd.read_csv("TQfree-1-repeat-2.csv",sep=",",skiprows=1).join(pd.read_csv('TQfree-2-repeat-2.csv',
                       skiprows=1).iloc[:,3:],rsuffix='2').join(pd.read_csv('TQfree-3-repeat-2.csv',
                       skiprows=1).iloc[:,3:],rsuffix='3')
    # cols 3 through 8 and 99 through 104 of the raw data are in standard buffer -- for the base spectra we use their average
    TQ_std_buf_avg_df = pd.DataFrame({"std_buf_avg":pd.concat([TQ_data_1.iloc[:,3:9],TQ_data_2.iloc[:,3:9],TQ_data_1.iloc[:,99:105],TQ_data_2.iloc[:,99:105]], axis=1).mean(axis=1)})
    TQ_6_std_buf_cols = TQ_std_buf_avg_df[['std_buf_avg','std_buf_avg','std_buf_avg','std_buf_avg','std_buf_avg','std_buf_avg']]
    # cols 195 through 200 of the raw data are in no-NaCl buffer -- for the base spectra we use their average
    TQ_no_nacl_buf_avg_df = pd.DataFrame({"no_nacl_buf_avg":pd.concat([TQ_data_1.iloc[:,195:201],TQ_data_2.iloc[:,195:201]], axis=1).mean(axis=1)})
    TQ_6_no_nacl_buf_cols = TQ_no_nacl_buf_avg_df[['no_nacl_buf_avg','no_nacl_buf_avg','no_nacl_buf_avg','no_nacl_buf_avg','no_nacl_buf_avg','no_nacl_buf_avg']]
    # get the averages for all solution conditions
    TQ_data_averaged = pd.concat([TQ_data_1, TQ_data_2]).groupby(level=0).mean()
    # build the data structure to send to get_efret:
    # - cols 0 through 2 are not data
    # - cols 3 through 8 are standard buffer (average of all repeats)
    # - cols 9 through 98 are averages for various solution conditions 
    # - cols 99 through 104 are standard buffer (average of all repeats)
    # - cols 105 through 194 are averages for various solution conditions 
    # - cols 195 through 200 are no-NaCl buffer (average of all repeats)
    # - cols 201 through 212 are averages for various solution conditions 
    TQ_data = pd.concat([TQ_data_1.iloc[:,:3],TQ_6_std_buf_cols,TQ_data_averaged.iloc[:,6:96], 
                        TQ_6_std_buf_cols,TQ_data_averaged.iloc[:,102:192],
                        TQ_6_no_nacl_buf_cols,TQ_data_averaged.iloc[:,198:]],axis=1)

    # Preprocess mNeonGreen base spectra in the same way
    NG_data_1=pd.read_csv("NGfree-1-repeat-1.csv",sep=",",skiprows=1).join(pd.read_csv('NGfree-2-repeat-1.csv',
                       skiprows=1).iloc[:,3:],rsuffix='2').join(pd.read_csv('NGfree-3-repeat-1.csv',
                       skiprows=1).iloc[:,3:],rsuffix='3')
    NG_data_2=pd.read_csv("NGfree-1-repeat-2.csv",sep=",",skiprows=1).join(pd.read_csv('NGfree-2-repeat-2.csv',
                       skiprows=1).iloc[:,3:],rsuffix='2').join(pd.read_csv('NGfree-3-repeat-2.csv',
                       skiprows=1).iloc[:,3:],rsuffix='3')
    NG_std_buf_avg_df = pd.DataFrame({"std_buf_avg":pd.concat([NG_data_1.iloc[:,3:9],NG_data_2.iloc[:,3:9],NG_data_1.iloc[:,99:105],NG_data_2.iloc[:,99:105]], axis=1).mean(axis=1)})
    NG_6_std_buf_cols = NG_std_buf_avg_df[['std_buf_avg','std_buf_avg','std_buf_avg','std_buf_avg','std_buf_avg','std_buf_avg']]
    NG_no_nacl_buf_avg_df = pd.DataFrame({"no_nacl_buf_avg":pd.concat([NG_data_1.iloc[:,195:201],NG_data_2.iloc[:,195:201]], axis=1).mean(axis=1)})
    NG_6_no_nacl_buf_cols = NG_no_nacl_buf_avg_df[['no_nacl_buf_avg','no_nacl_buf_avg','no_nacl_buf_avg','no_nacl_buf_avg','no_nacl_buf_avg','no_nacl_buf_avg']]
    NG_data_averaged = pd.concat([NG_data_1, NG_data_2]).groupby(level=0).mean()
    NG_data = pd.concat([NG_data_1.iloc[:,:3],NG_6_std_buf_cols,NG_data_averaged.iloc[:,6:96], 
                        NG_6_std_buf_cols,NG_data_averaged.iloc[:,102:192],
                        NG_6_no_nacl_buf_cols,NG_data_averaged.iloc[:,198:]],axis=1)                        
    return (TQ_data, NG_data)

############################################################
# Reject outliers for linear fit based on standard deviation
############################################################
def reject_outliers(data, m = 2):
    return(np.where(np.isin(data,(data[abs(data - np.mean(data)) < m * np.std(data)])))) # keep data within m (= 2) std dev of mean

#########################################################################################################
# Calculate average intensity / dir A for all IDRs in buffer. IDR fluorescence spectra will be normalized
# to this ratio so that correction for cross-excitation will be weighted the same for all IDRs.
#########################################################################################################
def get_avg_idr_dirA_intensity_ratio():
    NG_data_1=pd.read_csv("NGfree-1-repeat-1.csv",sep=",",skiprows=1).join(pd.read_csv('NGfree-2-repeat-1.csv',
                       skiprows=1).iloc[:,3:],rsuffix='2').join(pd.read_csv('NGfree-3-repeat-1.csv',
                       skiprows=1).iloc[:,3:],rsuffix='3')
    NG_data_2=pd.read_csv("NGfree-1-repeat-2.csv",sep=",",skiprows=1).join(pd.read_csv('NGfree-2-repeat-2.csv',
                       skiprows=1).iloc[:,3:],rsuffix='2').join(pd.read_csv('NGfree-3-repeat-2.csv',
                       skiprows=1).iloc[:,3:],rsuffix='3')
    NG_std_buf_avg_df = pd.DataFrame({"std_buf_avg":pd.concat([NG_data_1.iloc[:,3:9],NG_data_2.iloc[:,3:9],NG_data_1.iloc[:,99:105],NG_data_2.iloc[:,99:105]], axis=1).mean(axis=1)})

    all_csvs = ['GS0-1-repeat-1.csv','GS0-2-repeat-1.csv','GS0-3-repeat-1.csv',
                'GS0-1-repeat-2.csv','GS0-2-repeat-2.csv','GS0-3-repeat-2.csv',
                'GS8-1-repeat-1.csv','GS8-2-repeat-1.csv','GS8-3-repeat-1.csv',
                'GS8-1-repeat-2.csv','GS8-2-repeat-2.csv','GS8-3-repeat-2.csv',
                'GS16-1-repeat-1.csv','GS16-2-repeat-1.csv','GS16-3-repeat-1.csv',
                'GS16-1-repeat-2.csv','GS16-2-repeat-2.csv','GS16-3-repeat-2.csv',
                'GS24-1-repeat-1.csv','GS24-2-repeat-1.csv','GS24-3-repeat-1.csv',
                'GS24-1-repeat-2.csv','GS24-2-repeat-2.csv','GS24-3-repeat-2.csv',
                'GS32-1-repeat-1.csv','GS32-2-repeat-1.csv','GS32-3-repeat-1.csv',
                'GS32-1-repeat-2.csv','GS32-2-repeat-2.csv','GS32-3-repeat-2.csv',
                'GS48-1-repeat-1.csv','GS48-2-repeat-1.csv','GS48-3-repeat-1.csv',
                'GS48-1-repeat-2.csv','GS48-2-repeat-2.csv','GS48-3-repeat-2.csv',
                'puma-wt-1-repeat-1.csv','puma-wt-2-repeat-1.csv','puma-wt-3-repeat-1.csv',
                'puma-wt-1-repeat-2.csv','puma-wt-2-repeat-2.csv','puma-wt-3-repeat-2.csv',
                'puma-s1-1-repeat-1.csv','puma-s1-2-repeat-1.csv','puma-s1-3-repeat-1.csv',
                'puma-s1-1-repeat-2.csv','puma-s1-2-repeat-2.csv','puma-s1-3-repeat-2.csv',
                'puma-s2-1-repeat-1.csv','puma-s2-2-repeat-1.csv','puma-s2-3-repeat-1.csv',
                'puma-s2-1-repeat-2.csv','puma-s2-2-repeat-2.csv','puma-s2-3-repeat-2.csv',
                'puma-s3-1-repeat-1.csv','puma-s3-2-repeat-1.csv','puma-s3-3-repeat-1.csv',
                'puma-s3-1-repeat-2.csv','puma-s3-2-repeat-2.csv','puma-s3-3-repeat-2.csv',
                'ash1-1-repeat-1.csv','ash1-2-repeat-1.csv','ash1-3-repeat-1.csv',
                'ash1-1-repeat-2.csv','ash1-2-repeat-2.csv','ash1-3-repeat-2.csv',
                'e1a-1-repeat-1.csv','e1a-2-repeat-1.csv','e1a-3-repeat-1.csv',
                'e1a-1-repeat-2.csv','e1a-2-repeat-2.csv','e1a-3-repeat-2.csv',
                'p53-1-repeat-1.csv','p53-2-repeat-1.csv','p53-3-repeat-1.csv',
                'p53-1-repeat-2.csv','p53-2-repeat-2.csv','p53-3-repeat-2.csv',
                'fus-1-repeat-1.csv','fus-2-repeat-1.csv','fus-3-repeat-1.csv',
                'fus-1-repeat-2.csv','fus-2-repeat-2.csv','fus-3-repeat-2.csv']
    
    solutes = ['Buffer', 'Buffer.1', 'Buffer.2', 'Buffer.3', 'Buffer.4', 'Buffer.5']

    idr_to_direct_a_ratios = np.array([])
    for i in range(len(all_csvs)):
        df = (pd.read_csv(all_csvs[i], skiprows=1).iloc[:,2:])
        for j in range(len(solutes)):
            y = [int(x) for x in df[solutes[j]][1:].to_numpy()]
            idr_to_direct_a_ratios = np.append(idr_to_direct_a_ratios, sum(y) / NG_std_buf_avg_df.std_buf_avg.sum())
    avg_idr_dirA_intensity_ratio = idr_to_direct_a_ratios.mean()
    # print ('average_idr_dirA_intensity_ratio: ' + str(avg_idr_dirA_intensity_ratio))
    return (avg_idr_dirA_intensity_ratio)

######################################################################################################################
# Calculate FRET efficiency procedure (from Mastop et al Sci Rep 2017):    
#     1) F = f_raw - f_a
#        Subtract cross-excitation spectrum from acceptor only construct from full spectrum
#        f_raw is raw spectrum of FRET construct under donor excitation
#        f_a is acceptor only under donor excitation (obtained by testing mNeonGreen alone)
#        F is the corrected fluorescence spectrum
#     2) Linear regression of: F = F_d * f_d + F_s * f_a
#        f_d is donor only spectrum under donor excitation (obtained by testing mTurquoise2 alone)
#        F_d and F_s are fit parameters multiplying base spectra to obtain corrected FRET spectrum F
#     3) apparent fret efficiency E = 1 - (F_d / ((((Q_d*f_d) / (Q_a*f_a)) * F_s) + F_d))
#
# Arguments:
#     idr: name of IDR to be stored in Efret dataframe
#     df: dataframe containing raw fluorescence data for one IDR in many solutes
#     sel: specifies which columns in df (six columns per solute) to process
#     base_corr_fact_df: data frame containing correction factors for base spectra
#     TQ_data and NG_data: raw data for FRET donor and acceptor base spectra after processing by preprocess_tq_ng_data
#     correct_base_spectra: specifies whether to correct base spectra fluorescence intensities
######################################################################################################################
def get_efret(idr,df,sel,base_corr_fact_df,TQ_data,NG_data,correct_base_spectra,avg_idr_dirA_intensity_ratio):
    print ('get_efret')
    # print ('selecting ' + str(sel) + ' from dataframe')
    print ('idr: ' + idr)
    reg = LinearRegression(fit_intercept=False)
    fretRes=pd.DataFrame(columns=['col','sol','conc','E_f'])
    Q_d=0.93 # Mastop et al Sci Rep 2017
    Q_a=0.8  # Mastop et al Sci Rep 2017 
    # R_naught=62 # Angstroms, Mastop et al Sci Rep 2017 
    
    # Get wavelengths (integer values only) for x axis
    x_range = df.iloc[1:,2].astype('int64')
    
    # Iterate through the selected columns - each column has a spectrum for one IDR in one solution condition
    for col in sel:
        # Get the correction factors for the base spectra - each column is one solution condition
        try:
            f_a_factor=base_corr_fact_df[base_corr_fact_df['col']==col]['f_a_corr'].iloc[0] 
            f_d_factor=base_corr_fact_df[base_corr_fact_df['col']==col]['f_d_corr'].iloc[0] 
        except:
            f_a_factor=1
            f_d_factor=1
            print("%i factor not found"%col)
        # Get the uncorrected raw data for the base spectra
        NG_raw = NG_data.iloc[1:,col]
        TQ_raw = TQ_data.iloc[1:,col] 
        
        if correct_base_spectra:
            # Apply correction factors (this is what we did in the paper)
            NG_corrected = NG_raw*f_a_factor 
            TQ_corrected = TQ_raw*f_d_factor 
        else:
            # Don't apply correction factors
            NG_corrected = NG_raw 
            TQ_corrected = TQ_raw 
        
        # Normalize both spectra by donor peak so as to be able to compare with Mastop 2017, Fig. 4 
        # (End results are not affected if this is omitted)
        # f_a is the acceptor base spectrum
        # f_d is the donor base spectrum
        f_a = 100 * NG_corrected / TQ_corrected[26]
        f_d = 100 * TQ_corrected / TQ_corrected[26]
        
        # Get the name of the solute for the current column
        sol = df.columns[col]
        # Sanity check for column headers
        sol_a = NG_data.columns[col]
        sol_d = TQ_data.columns[col]
        if (sol != sol_a) | (sol != sol_d):
            if not ('buf' in sol_a and 'buf' in sol_d and 'Buf' in sol):
                print(sol,sol_a,sol_d)
                print('Mismatch in solution type!')
                return()
        # Get the solute concentration relevant to the current column
        conc = df.iloc[0,col]
        # Join donor-only and acceptor-only spectra side-by-side to send to linear regression function
        X=pd.concat([f_d,f_a],axis=1) 
        # Get the raw spectrum before correction for cross-excitation
        y_raw = df.iloc[1:,col] 
        # Normalize raw spectrum so that all experimental spectra have the same IDR/dirA ratio.
        if (sum(y_raw) > 0):
            y_raw_idr_dirA_ratio = y_raw.sum() / NG_corrected.sum()
            y_normalized = y_raw * (avg_idr_dirA_intensity_ratio / y_raw_idr_dirA_ratio)
        # Subtract corrected acceptor-only spectrum (i.e., cross-excitation) from raw data
        # y is the corrected spectrum to send to the linear regression function
        y = (y_normalized - NG_corrected) 
        
        # Send the experimental spectrum and the base spectra to the linear regression function reg.fit
        # It will return coefficients representing the donor and acceptor contributions to the experimental spectrum
        try:
            reg.fit(X, y)
            # fitResult is the fitted spectrum that is a linear combination of the base spectra
            fitResult=np.sum(X*reg.coef_,axis=1) 
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
        except:
            print ('getFret: reg fit did not work!')
            E_f=np.nan
            F_d=np.nan
            F_s=np.nan
            fitResult=np.nan

        # Append the results for this solution condition to the fretRes data frame                        
        fretRes= fretRes.append({'idr':idr,'col':col,'sol':sol,'conc':conc,'E_f':E_f,
                                 'E_f_all':E_f_all_lambda,
                                 'f_d':f_d,'f_a':f_a,'F_d':F_d.to_numpy(),'F_s':F_s.to_numpy(),'exp':y,
                                 'x_range':x_range,'fit':fitResult}, ignore_index=True)
    return (fretRes)
    
#######################################################################################################
# get_d_over_a: a mini version of get_efret which assumes d is from 450 to 489 and a is from 490 to 600  
#######################################################################################################
def get_d_over_a(idr,df,sel):
    print ('get_d_over_a')
    print ('idr: ' + idr)
    d_over_a_df = pd.DataFrame(columns=['col','sol','conc','d','a','d_over_a','a_over_d'])
    for col in sel:
        # print (str(df.iloc[:,col]))
        sol = df.columns[col]
        conc = df.iloc[0,col]
        d = sum(df.iloc[1:40,col])
        a = sum(df.iloc[41:,col])
        if a > 0:
            d_over_a = d/a
        else:
            d_over_a = None
        if d > 0:
            a_over_d = a/d
        else:
            a_over_d = None
        d_over_a_df = d_over_a_df.append({'idr':idr,'col':col,'sol':sol,'conc':conc,'d':d,'a':a,'d_over_a':d_over_a,'a_over_d':a_over_d}, ignore_index=True)
    return d_over_a_df
        
    
####################################################################################################################
# Fit measured spectra for free donor and acceptor and correct for differences in overall intensity
# Data points for a series of concentrations of a single solute are sent to linear regression function
# Data frame corr_fact_df with correction factors for all solution conditions is returned
# Correction factors will be applied in get_efret
#
# Arguments: 
# - TQ_data and NG_data: raw data for FRET donor and acceptor base spectra after processing by preprocess_tq_ng_data
####################################################################################################################
def build_base_correction_factor_df(TQ_data, NG_data):
    df=[TQ_data,NG_data] # df is a list of the two data frames TQ_data and NG_data
    # Create correction factor data frame
    corr_fact_df=pd.DataFrame(columns=['col','sol','conc','f_d_corr','f_a_corr'])
    # The first two columns are not data, so we start at column 3
    bufStart=3
    bufIdx=range(bufStart,bufStart+6) # bufIdx is a list of ints: the six buffer cols for TQ_data
    x=df[0].iloc[11:-20,2].astype('float64') # set up x axis to go from 460 to 580 nm, including only integer values (not x.5 values), so 121 points
    # A new data series (for a new solute) starts every six columns
    # Replicates in buffer will not be corrected -- fit will fail and factor of 1 will be assigned
    for val in range(3,212,6): # for the different column header values
        pattern = TQ_data.columns[val]
        selStart=val # int for first of six cols of current header
        selIdx=range(selStart,selStart+6) # list of ints for the six cols for the current header
        wlIndex=[25,68] # indices for 474 and 517 nm (TQ and NG emission peaks)
        corr_factor=np.ones([2,6]) # initialize correction factor matrix -- 6 factors, 0 is TQ, 1 is NG
        for i in range(2): # 0 is TQ_data, 1 is NG_data
            x_F=np.array(0.0)
            # Get the mean of the donor and acceptor peaks in buffer (i.e., solute concentration = 0)
            y_F=np.array(df[i].iloc[wlIndex[i],bufIdx].mean())
            for idx in selIdx:
                x_F=np.append(x_F,df[i].iloc[0,idx]) # concentrations from 0 to 24
                y_F=np.append(y_F,df[i].iloc[wlIndex[i],idx]) # fluorescence intensity
            try:
                fitIdx = reject_outliers(y_F,2) # eliminate data points where y > 2 std dev from mean
                with np.errstate(divide='ignore', invalid='ignore'):
                    pf = np.polyfit(x_F[fitIdx],y_F[fitIdx],1) # degree of polynomial fit is 1, so linear; just fitting the non-outliers
                y_F_corr=x_F*pf[0]+y_F[0] # y = mx + b, pf[0] is the slope
                fitText="slope: %.2f; int: %i" % (pf[0],pf[1]) # text for use in plots
                corr_factor[i,:]=(y_F_corr/y_F)[1:] # corr_factor[0] TQ corr factors (f_d_corr); corr_factor [1] NG corr factors (f_a_corr) 
            except:
                if not ('buf' in pattern):
                    print("Fit didn't work for sol %s" % pattern)
        # Add correction factors to "corr_fact_df" data frame
        for i in range(len(selIdx)):        
            corr_fact_df=corr_fact_df.append({'col':selIdx[i],'sol':NG_data.iloc[:,selIdx[i]].name,
                              'conc':NG_data.iloc[0,selIdx[i]],
                              'f_d_corr':corr_factor[0,i],'f_a_corr':corr_factor[1,i]}, 
                                ignore_index=True)
    return(corr_fact_df)
    
    

###################################################################################################################################
# Send average E_fret in standard buffer (20 mM NaPi, 100 mM NaCl) of GS linkers of various lengths to a linear regression function
# Return the slope and y-intercept of the line
###################################################################################################################################
def get_gs_linker_regression_in_buffer(GS0_efret_1, GS0_efret_2, GS8_efret_1, GS8_efret_2, GS16_efret_1, GS16_efret_2, GS24_efret_1, GS24_efret_2, GS32_efret_1, GS32_efret_2, GS48_efret_1, GS48_efret_2):   
    # X values for GS linkers and other IDRs
    linker_Ns = np.array([0,16,32,48,64,96])
    linker_df = pd.DataFrame({ "GS0":pd.concat([GS0_efret_1[GS0_efret_1['sol'].isin(['Buffer','Buffer.1','Buffer.2','Buffer.3','Buffer.4','Buffer.5',
                                                             'Buffer2','Buffer.12','Buffer.22','Buffer.32','Buffer.42','Buffer.52'])]['E_f'],
                                        GS0_efret_2[GS0_efret_2['sol'].isin(['Buffer','Buffer.1','Buffer.2','Buffer.3','Buffer.4','Buffer.5',
                                                             'Buffer2','Buffer.12','Buffer.22','Buffer.32','Buffer.42','Buffer.52'])]['E_f']], 
                                      axis=0, ignore_index=True),
                      "GS8":pd.concat([GS8_efret_1[GS8_efret_1['sol'].isin(['Buffer','Buffer.1','Buffer.2','Buffer.3','Buffer.4','Buffer.5',
                                                             'Buffer2','Buffer.12','Buffer.22','Buffer.32','Buffer.42','Buffer.52'])]['E_f'],
                                        GS8_efret_2[GS8_efret_2['sol'].isin(['Buffer','Buffer.1','Buffer.2','Buffer.3','Buffer.4','Buffer.5',
                                                             'Buffer2','Buffer.12','Buffer.22','Buffer.32','Buffer.42','Buffer.52'])]['E_f']], 
                                      axis=0, ignore_index=True),
                      "GS16":pd.concat([GS16_efret_1[GS16_efret_1['sol'].isin(['Buffer','Buffer.1','Buffer.2','Buffer.3','Buffer.4','Buffer.5',
                                                             'Buffer2','Buffer.12','Buffer.22','Buffer.32','Buffer.42','Buffer.52'])]['E_f'],
                                        GS16_efret_2[GS16_efret_2['sol'].isin(['Buffer','Buffer.1','Buffer.2','Buffer.3','Buffer.4','Buffer.5',
                                                             'Buffer2','Buffer.12','Buffer.22','Buffer.32','Buffer.42','Buffer.52'])]['E_f']], 
                                        axis=0, ignore_index=True),
                      "GS24":pd.concat([GS24_efret_1[GS24_efret_1['sol'].isin(['Buffer','Buffer.1','Buffer.2','Buffer.3','Buffer.4','Buffer.5',
                                                             'Buffer2','Buffer.12','Buffer.22','Buffer.32','Buffer.42','Buffer.52'])]['E_f'],
                                        GS24_efret_2[GS24_efret_2['sol'].isin(['Buffer','Buffer.1','Buffer.2','Buffer.3','Buffer.4','Buffer.5',
                                                             'Buffer2','Buffer.12','Buffer.22','Buffer.32','Buffer.42','Buffer.52'])]['E_f']], 
                                        axis=0, ignore_index=True), 
                      "GS32":pd.concat([GS32_efret_1[GS32_efret_1['sol'].isin(['Buffer','Buffer.1','Buffer.2','Buffer.3','Buffer.4','Buffer.5',
                                                             'Buffer2','Buffer.12','Buffer.22','Buffer.32','Buffer.42','Buffer.52'])]['E_f'],
                                        GS32_efret_2[GS32_efret_2['sol'].isin(['Buffer','Buffer.1','Buffer.2','Buffer.3','Buffer.4','Buffer.5',
                                                             'Buffer2','Buffer.12','Buffer.22','Buffer.32','Buffer.42','Buffer.52'])]['E_f']], 
                                        axis=0, ignore_index=True), 
                      "GS48":pd.concat([GS48_efret_1[GS48_efret_1['sol'].isin(['Buffer','Buffer.1','Buffer.2','Buffer.3','Buffer.4','Buffer.5',
                                                             'Buffer2','Buffer.12','Buffer.22','Buffer.32','Buffer.42','Buffer.52'])]['E_f'],
                                        GS48_efret_2[GS48_efret_2['sol'].isin(['Buffer','Buffer.1','Buffer.2','Buffer.3','Buffer.4','Buffer.5',
                                                             'Buffer2','Buffer.12','Buffer.22','Buffer.32','Buffer.42','Buffer.52'])]['E_f']], 
                                        axis=0, ignore_index=True) })
    # Get average Efret in buffer for GS linkers
    mean_linker_Efrets = linker_df.mean(axis=0)
    # After linear regression of GS Efret values, Efret(GS) = slope * N + intercept 
    slope, intercept, r_value, p_value, std_err = stats.linregress(linker_Ns, mean_linker_Efrets)
    return slope, intercept
    
###########################################################################################################################################
# convert_efret_df_to_chi_df: convert E_fret data for an IDR to a dataframe containing chi values of the IDR in different solute conditions
#
# Arguments:
# - idr: the IDR
# - df: a dataframe produced by the get_efret function
# - gs_regress_slope: the slope returned by get_gs_linker_regression_in_buffer
# - gs_regress_intercept: the y-intercept returned by get_gs_linker_regression_in_buffer
# - N: length in amino acids of the IDR
# - repeat: experimental repeat number for the IDR
# - solutes: list of solutes in which we want to calculate chi for the IDR
###########################################################################################################################################
def convert_efret_df_to_chi_df(idr, df, gs_regress_slope, gs_regress_intercept, N, repeat, solutes):
    chi_df=pd.DataFrame(columns=['idr','N','repeat','sol','conc','unit','E_f','E_f_GS','chi'])
    num_rows = df.shape[0]
    n_solute = 1.33
    n_buffer = 1.33
    implied_efret_gs = gs_regress_slope * N + gs_regress_intercept
    for i in range(0, num_rows//6):
        start_idx = i * 6
        solute = df['sol'][start_idx]
        efret_series = df['E_f'][start_idx:start_idx + 6].to_numpy()
        if solute[0:7] == 'Buffer3':
            gs_normalized_rees = (n_solute * (((1/efret_series)-1)**(1/6))) / (n_buffer * (((1/implied_efret_gs)-1)**(1/6)))
            chis = gs_normalized_rees - 1
            curr_buffer_avg_chi = chis.mean()
            for i in range(len(chis)):
                chi_df = chi_df.append({'idr':idr,'N':int(N),'repeat':repeat,'sol':'Buffer (no NaCl)','conc':0,'unit':'N/A',
                                        'E_f':efret_series[i],'E_f_GS':implied_efret_gs,'chi':chis[i]}, ignore_index=True)
        elif solute[0:6] == 'Buffer':
            gs_normalized_rees = (n_solute * (((1/efret_series)-1)**(1/6))) / (n_buffer * (((1/implied_efret_gs)-1)**(1/6)))
            chis = gs_normalized_rees - 1
            curr_buffer_avg_chi = chis.mean()
            for i in range(len(chis)):
                chi_df = chi_df.append({'idr':idr,'N':int(N),'repeat':repeat,'sol':'Buffer (standard)','conc':0,'unit':'N/A',
                                        'E_f':efret_series[i],'E_f_GS':implied_efret_gs,'chi':chis[i]}, ignore_index=True)
        else:
            if solute in solutes:
                # unit
                if solute in ['Urea', 'GuHCl', 'NaCl', 'KCl']:
                    unit = 'M'
                else:
                    unit = 'conc'
                # concentrations
                concs = df['conc'][start_idx:start_idx + 6].to_numpy()
                # chis
                # For points that fall off the chart, perhaps due to semidilute regime
                for i in range(len(efret_series)):
                    if efret_series[i] >= 1:
                        efret_series[i] = 0.999999
                gs_normalized_rees = (n_solute * (((1/efret_series)-1)**(1/6))) / (n_buffer * (((1/implied_efret_gs)-1)**(1/6)))
                chis = gs_normalized_rees - 1
                chi_df = chi_df.append({'idr':idr,'N':int(N),'repeat':repeat,'sol':solute,'conc':0,'unit':unit,'E_f':'(average)',
                                        'E_f_GS':'(average)','chi':curr_buffer_avg_chi}, ignore_index=True)
                for i in range(len(chis)):
                    chi_df = chi_df.append({'idr':idr,'N':int(N),'repeat':repeat,'sol':solute,'conc':concs[i],'unit':unit,
                                            'E_f':efret_series[i],'E_f_GS':implied_efret_gs,'chi':chis[i]}, ignore_index=True)
    return (chi_df)
    
    
####################################################################################################
# get_all_idrs_chi_df:
# - reads in experimental data from csv files 
# - calls get_efret to convert raw fluorescence spectra to FRET efficiency
# - calls convert_efret_df_to_chi_df to convert FRET efficiency into chi
# - returns a dataframe containing chi values for all repeats of all IDRs in all solution conditions
####################################################################################################
def get_all_idrs_chi_df():
    # Read in and preprocess raw data for mTurquoise2 and mNeonGreen base spectra
    TQ_data, NG_data = preprocess_tq_ng_data()

    # Read in raw data for IDRs
    GS0_data_1=pd.read_csv("GS0-1-repeat-1.csv",sep=",",skiprows=1).join(pd.read_csv('GS0-2-repeat-1.csv',
                        skiprows=1).iloc[:,3:],rsuffix='2').join(pd.read_csv('GS0-3-repeat-1.csv',
                        skiprows=1).iloc[:,3:],rsuffix='3')
    GS0_data_2=pd.read_csv("GS0-1-repeat-2.csv",sep=",",skiprows=1).join(pd.read_csv('GS0-2-repeat-2.csv',
                        skiprows=1).iloc[:,3:],rsuffix='2').join(pd.read_csv('GS0-3-repeat-2.csv',
                        skiprows=1).iloc[:,3:],rsuffix='3')
    GS8_data_1=pd.read_csv("GS8-1-repeat-1.csv",sep=",",skiprows=1).join(pd.read_csv('GS8-2-repeat-1.csv',
                        skiprows=1).iloc[:,3:],rsuffix='2').join(pd.read_csv('GS8-3-repeat-1.csv',
                        skiprows=1).iloc[:,3:],rsuffix='3')
    GS8_data_2=pd.read_csv("GS8-1-repeat-2.csv",sep=",",skiprows=1).join(pd.read_csv('GS8-2-repeat-2.csv',
                        skiprows=1).iloc[:,3:],rsuffix='2').join(pd.read_csv('GS8-3-repeat-2.csv',
                        skiprows=1).iloc[:,3:],rsuffix='3')
    GS16_data_1=pd.read_csv("GS16-1-repeat-1.csv",sep=",",skiprows=1).join(pd.read_csv('GS16-2-repeat-1.csv',
                        skiprows=1).iloc[:,3:],rsuffix='2').join(pd.read_csv('GS16-3-repeat-1.csv',
                        skiprows=1).iloc[:,3:],rsuffix='3')
    GS16_data_2=pd.read_csv("GS16-1-repeat-2.csv",sep=",",skiprows=1).join(pd.read_csv('GS16-2-repeat-2.csv',
                        skiprows=1).iloc[:,3:],rsuffix='2').join(pd.read_csv('GS16-3-repeat-2.csv',
                        skiprows=1).iloc[:,3:],rsuffix='3')
    GS24_data_1=pd.read_csv("GS24-1-repeat-1.csv",sep=",",skiprows=1).join(pd.read_csv('GS24-2-repeat-1.csv',
                        skiprows=1).iloc[:,3:],rsuffix='2').join(pd.read_csv('GS24-3-repeat-1.csv',
                        skiprows=1).iloc[:,3:],rsuffix='3')
    GS24_data_2=pd.read_csv("GS24-1-repeat-2.csv",sep=",",skiprows=1).join(pd.read_csv('GS24-2-repeat-2.csv',
                        skiprows=1).iloc[:,3:],rsuffix='2').join(pd.read_csv('GS24-3-repeat-2.csv',
                        skiprows=1).iloc[:,3:],rsuffix='3')
    GS32_data_1=pd.read_csv("GS32-1-repeat-1.csv",sep=",",skiprows=1).join(pd.read_csv('GS32-2-repeat-1.csv',
                        skiprows=1).iloc[:,3:],rsuffix='2').join(pd.read_csv('GS32-3-repeat-1.csv',
                        skiprows=1).iloc[:,3:],rsuffix='3')
    GS32_data_2=pd.read_csv("GS32-1-repeat-2.csv",sep=",",skiprows=1).join(pd.read_csv('GS32-2-repeat-2.csv',
                        skiprows=1).iloc[:,3:],rsuffix='2').join(pd.read_csv('GS32-3-repeat-2.csv',
                        skiprows=1).iloc[:,3:],rsuffix='3')
    GS48_data_1=pd.read_csv("GS48-1-repeat-1.csv",sep=",",skiprows=1).join(pd.read_csv('GS48-2-repeat-1.csv',
                        skiprows=1).iloc[:,3:],rsuffix='2').join(pd.read_csv('GS48-3-repeat-1.csv',
                        skiprows=1).iloc[:,3:],rsuffix='3')
    GS48_data_2=pd.read_csv("GS48-1-repeat-2.csv",sep=",",skiprows=1).join(pd.read_csv('GS48-2-repeat-2.csv',
                        skiprows=1).iloc[:,3:],rsuffix='2').join(pd.read_csv('GS48-3-repeat-2.csv',
                        skiprows=1).iloc[:,3:],rsuffix='3')
    PUMA_data_1=pd.read_csv("puma-wt-1-repeat-1.csv",sep=",",skiprows=1).join(pd.read_csv('puma-wt-2-repeat-1.csv',
                        skiprows=1).iloc[:,3:],rsuffix='2').join(pd.read_csv('puma-wt-3-repeat-1.csv',
                        skiprows=1).iloc[:,3:],rsuffix='3')
    PUMA_data_2=pd.read_csv("puma-wt-1-repeat-2.csv",sep=",",skiprows=1).join(pd.read_csv('puma-wt-2-repeat-2.csv',
                        skiprows=1).iloc[:,3:],rsuffix='2').join(pd.read_csv('puma-wt-3-repeat-2.csv',
                        skiprows=1).iloc[:,3:],rsuffix='3')
    PUMAS1_data_1=pd.read_csv("puma-s1-1-repeat-1.csv",sep=",",skiprows=1).join(pd.read_csv('puma-s1-2-repeat-1.csv',
                        skiprows=1).iloc[:,3:],rsuffix='2').join(pd.read_csv('puma-s1-3-repeat-1.csv',
                        skiprows=1).iloc[:,3:],rsuffix='3')
    PUMAS1_data_2=pd.read_csv("puma-s1-1-repeat-2.csv",sep=",",skiprows=1).join(pd.read_csv('puma-s1-2-repeat-2.csv',
                        skiprows=1).iloc[:,3:],rsuffix='2').join(pd.read_csv('puma-s1-3-repeat-2.csv',
                        skiprows=1).iloc[:,3:],rsuffix='3')
    PUMAS2_data_1=pd.read_csv("puma-s2-1-repeat-1.csv",sep=",",skiprows=1).join(pd.read_csv('puma-s2-2-repeat-1.csv',
                        skiprows=1).iloc[:,3:],rsuffix='2').join(pd.read_csv('puma-s2-3-repeat-1.csv',
                        skiprows=1).iloc[:,3:],rsuffix='3')
    PUMAS2_data_2=pd.read_csv("puma-s2-1-repeat-2.csv",sep=",",skiprows=1).join(pd.read_csv('puma-s2-2-repeat-2.csv',
                        skiprows=1).iloc[:,3:],rsuffix='2').join(pd.read_csv('puma-s2-3-repeat-2.csv',
                        skiprows=1).iloc[:,3:],rsuffix='3')
    PUMAS3_data_1=pd.read_csv("puma-s3-1-repeat-1.csv",sep=",",skiprows=1).join(pd.read_csv('puma-s3-2-repeat-1.csv',
                        skiprows=1).iloc[:,3:],rsuffix='2').join(pd.read_csv('puma-s3-3-repeat-1.csv',
                        skiprows=1).iloc[:,3:],rsuffix='3')
    PUMAS3_data_2=pd.read_csv("puma-s3-1-repeat-2.csv",sep=",",skiprows=1).join(pd.read_csv('puma-s3-2-repeat-2.csv',
                        skiprows=1).iloc[:,3:],rsuffix='2').join(pd.read_csv('puma-s3-3-repeat-2.csv',
                        skiprows=1).iloc[:,3:],rsuffix='3')
    Ash1_data_1=pd.read_csv("ash1-1-repeat-1.csv",sep=",",skiprows=1).join(pd.read_csv('ash1-2-repeat-1.csv',
                        skiprows=1).iloc[:,3:],rsuffix='2').join(pd.read_csv('ash1-3-repeat-1.csv',
                        skiprows=1).iloc[:,3:],rsuffix='3')
    Ash1_data_2=pd.read_csv("ash1-1-repeat-2.csv",sep=",",skiprows=1).join(pd.read_csv('ash1-2-repeat-2.csv',
                        skiprows=1).iloc[:,3:],rsuffix='2').join(pd.read_csv('ash1-3-repeat-2.csv',
                        skiprows=1).iloc[:,3:],rsuffix='3')
    E1A_data_1=pd.read_csv("e1a-1-repeat-1.csv",sep=",",skiprows=1).join(pd.read_csv('e1a-2-repeat-1.csv',
                        skiprows=1).iloc[:,3:],rsuffix='2').join(pd.read_csv('e1a-3-repeat-1.csv',
                        skiprows=1).iloc[:,3:],rsuffix='3')
    E1A_data_2=pd.read_csv("e1a-1-repeat-2.csv",sep=",",skiprows=1).join(pd.read_csv('e1a-2-repeat-2.csv',
                        skiprows=1).iloc[:,3:],rsuffix='2').join(pd.read_csv('e1a-3-repeat-2.csv',
                        skiprows=1).iloc[:,3:],rsuffix='3')
    P53_data_1=pd.read_csv("p53-1-repeat-1.csv",sep=",",skiprows=1).join(pd.read_csv('p53-2-repeat-1.csv',
                        skiprows=1).iloc[:,3:],rsuffix='2').join(pd.read_csv('p53-3-repeat-1.csv',
                        skiprows=1).iloc[:,3:],rsuffix='3')
    P53_data_2=pd.read_csv("p53-1-repeat-2.csv",sep=",",skiprows=1).join(pd.read_csv('p53-2-repeat-2.csv',
                        skiprows=1).iloc[:,3:],rsuffix='2').join(pd.read_csv('p53-3-repeat-2.csv',
                        skiprows=1).iloc[:,3:],rsuffix='3')
    FUS_data_1=pd.read_csv("fus-1-1-test.csv",sep=",",skiprows=1).join(pd.read_csv('fus-2-repeat-1.csv',
                        skiprows=1).iloc[:,3:],rsuffix='2').join(pd.read_csv('fus-3-repeat-1.csv',
                        skiprows=1).iloc[:,3:],rsuffix='3')
    FUS_data_2=pd.read_csv("fus-1-repeat-2.csv",sep=",",skiprows=1).join(pd.read_csv('fus-2-repeat-2.csv',
                        skiprows=1).iloc[:,3:],rsuffix='2').join(pd.read_csv('fus-3-repeat-2.csv',
                        skiprows=1).iloc[:,3:],rsuffix='3')

    # Build a data frame with correction factors for the donor and acceptor at each concentration of each solute
    base_corr_fact_df=build_base_correction_factor_df(TQ_data, NG_data)
    
    # Get the average IDR/dirA intensity ratio for normalization of experimental spectra
    avg_idr_dirA_intensity_ratio = get_avg_idr_dirA_intensity_ratio()

    # Build an E_fret data frame for each repeat for each IDR
    GS0_efret_1=get_efret('GS0',GS0_data_1,range(3,GS0_data_1.shape[1]),base_corr_fact_df,TQ_data,NG_data,True,avg_idr_dirA_intensity_ratio)
    GS0_efret_2=get_efret('GS0',GS0_data_2,range(3,GS0_data_2.shape[1]),base_corr_fact_df,TQ_data,NG_data,True,avg_idr_dirA_intensity_ratio)
    GS8_efret_1=get_efret('GS8',GS8_data_1,range(3,GS8_data_1.shape[1]),base_corr_fact_df,TQ_data,NG_data,True,avg_idr_dirA_intensity_ratio)
    GS8_efret_2=get_efret('GS8',GS8_data_2,range(3,GS8_data_2.shape[1]),base_corr_fact_df,TQ_data,NG_data,True,avg_idr_dirA_intensity_ratio)
    GS16_efret_1=get_efret('GS16',GS16_data_1,range(3,GS16_data_1.shape[1]),base_corr_fact_df,TQ_data,NG_data,True,avg_idr_dirA_intensity_ratio)
    GS16_efret_2=get_efret('GS16',GS16_data_2,range(3,GS16_data_2.shape[1]),base_corr_fact_df,TQ_data,NG_data,True,avg_idr_dirA_intensity_ratio)
    GS24_efret_1=get_efret('GS24',GS24_data_1,range(3,GS24_data_1.shape[1]),base_corr_fact_df,TQ_data,NG_data,True,avg_idr_dirA_intensity_ratio)
    GS24_efret_2=get_efret('GS24',GS24_data_2,range(3,GS24_data_2.shape[1]),base_corr_fact_df,TQ_data,NG_data,True,avg_idr_dirA_intensity_ratio)
    GS32_efret_1=get_efret('GS32',GS32_data_1,range(3,GS32_data_1.shape[1]),base_corr_fact_df,TQ_data,NG_data,True,avg_idr_dirA_intensity_ratio)
    GS32_efret_2=get_efret('GS32',GS32_data_2,range(3,GS32_data_2.shape[1]),base_corr_fact_df,TQ_data,NG_data,True,avg_idr_dirA_intensity_ratio)
    GS48_efret_1=get_efret('GS48',GS48_data_1,range(3,GS48_data_1.shape[1]),base_corr_fact_df,TQ_data,NG_data,True,avg_idr_dirA_intensity_ratio)
    GS48_efret_2=get_efret('GS48',GS48_data_2,range(3,GS48_data_2.shape[1]),base_corr_fact_df,TQ_data,NG_data,True,avg_idr_dirA_intensity_ratio)
    PUMA_efret_1=get_efret('PUMA WT',PUMA_data_1,range(3,PUMA_data_1.shape[1]),base_corr_fact_df,TQ_data,NG_data,True,avg_idr_dirA_intensity_ratio)
    PUMA_efret_2=get_efret('PUMA WT',PUMA_data_2,range(3,PUMA_data_2.shape[1]),base_corr_fact_df,TQ_data,NG_data,True,avg_idr_dirA_intensity_ratio)
    PUMAS1_efret_1=get_efret('PUMA S1',PUMAS1_data_1,range(3,PUMAS1_data_1.shape[1]),base_corr_fact_df,TQ_data,NG_data,True,avg_idr_dirA_intensity_ratio)
    PUMAS1_efret_2=get_efret('PUMA S1',PUMAS1_data_2,range(3,PUMAS1_data_2.shape[1]),base_corr_fact_df,TQ_data,NG_data,True,avg_idr_dirA_intensity_ratio)
    PUMAS2_efret_1=get_efret('PUMA S2',PUMAS2_data_1,range(3,PUMAS2_data_1.shape[1]),base_corr_fact_df,TQ_data,NG_data,True,avg_idr_dirA_intensity_ratio)
    PUMAS2_efret_2=get_efret('PUMA S2',PUMAS2_data_2,range(3,PUMAS2_data_2.shape[1]),base_corr_fact_df,TQ_data,NG_data,True,avg_idr_dirA_intensity_ratio)
    PUMAS3_efret_1=get_efret('PUMA S3',PUMAS3_data_1,range(3,PUMAS3_data_1.shape[1]),base_corr_fact_df,TQ_data,NG_data,True,avg_idr_dirA_intensity_ratio)
    PUMAS3_efret_2=get_efret('PUMA S3',PUMAS3_data_2,range(3,PUMAS3_data_2.shape[1]),base_corr_fact_df,TQ_data,NG_data,True,avg_idr_dirA_intensity_ratio)
    Ash1_efret_1=get_efret('Ash1',Ash1_data_1,range(3,Ash1_data_1.shape[1]),base_corr_fact_df,TQ_data,NG_data,True,avg_idr_dirA_intensity_ratio)
    Ash1_efret_2=get_efret('Ash1',Ash1_data_2,range(3,Ash1_data_2.shape[1]),base_corr_fact_df,TQ_data,NG_data,True,avg_idr_dirA_intensity_ratio)
    E1A_efret_1=get_efret('E1A',E1A_data_1,range(3,E1A_data_1.shape[1]),base_corr_fact_df,TQ_data,NG_data,True,avg_idr_dirA_intensity_ratio)
    E1A_efret_2=get_efret('E1A',E1A_data_2,range(3,E1A_data_2.shape[1]),base_corr_fact_df,TQ_data,NG_data,True,avg_idr_dirA_intensity_ratio)
    P53_efret_1=get_efret('p53',P53_data_1,range(3,P53_data_1.shape[1]),base_corr_fact_df,TQ_data,NG_data,True,avg_idr_dirA_intensity_ratio)
    P53_efret_2=get_efret('p53',P53_data_2,range(3,P53_data_2.shape[1]),base_corr_fact_df,TQ_data,NG_data,True,avg_idr_dirA_intensity_ratio)
    FUS_efret_1=get_efret('FUS',FUS_data_1,range(3,FUS_data_1.shape[1]),base_corr_fact_df,TQ_data,NG_data,True,avg_idr_dirA_intensity_ratio)
    FUS_efret_2=get_efret('FUS',FUS_data_2,range(3,FUS_data_1.shape[1]),base_corr_fact_df,TQ_data,NG_data,True,avg_idr_dirA_intensity_ratio)

    print ('E_fret values computed!')
    
    # Get the regression line for the GS linkers in buffer
    gs_regress_slope, gs_regress_intercept = get_gs_linker_regression_in_buffer(GS0_efret_1, GS0_efret_2, GS8_efret_1, GS8_efret_2, GS16_efret_1, GS16_efret_2, GS24_efret_1, GS24_efret_2, GS32_efret_1, GS32_efret_2, GS48_efret_1, GS48_efret_2)

    print ('Computing chi values...')

    # List of co-solutes used in the hidden structure project.
    solutes = ['PEG2000', 'Ficoll', 'Glycine', 'Sarcosine', 'Urea', 'GuHCl', 'NaCl', 'KCl']
    
    # Compute chi for each IDR in each solution condition
    GS0_chi_df_1 = convert_efret_df_to_chi_df('GS0', GS0_efret_1, gs_regress_slope, gs_regress_intercept, 0, 1, solutes)
    GS8_chi_df_1 = convert_efret_df_to_chi_df('GS8', GS8_efret_1, gs_regress_slope, gs_regress_intercept, 16, 1, solutes)
    GS16_chi_df_1 = convert_efret_df_to_chi_df('GS16', GS16_efret_1, gs_regress_slope, gs_regress_intercept, 32, 1, solutes)
    GS24_chi_df_1 = convert_efret_df_to_chi_df('GS24', GS24_efret_1, gs_regress_slope, gs_regress_intercept, 48, 1, solutes)
    GS32_chi_df_1 = convert_efret_df_to_chi_df('GS32', GS32_efret_1, gs_regress_slope, gs_regress_intercept, 64, 1, solutes)
    GS48_chi_df_1 = convert_efret_df_to_chi_df('GS48', GS48_efret_1, gs_regress_slope, gs_regress_intercept, 96, 1, solutes)
    PUMA_chi_df_1 = convert_efret_df_to_chi_df('PUMA WT', PUMA_efret_1, gs_regress_slope, gs_regress_intercept, 34, 1, solutes)
    PUMAS1_chi_df_1 = convert_efret_df_to_chi_df('PUMA S1', PUMAS1_efret_1, gs_regress_slope, gs_regress_intercept, 34, 1, solutes)
    PUMAS2_chi_df_1 = convert_efret_df_to_chi_df('PUMA S2', PUMAS2_efret_1, gs_regress_slope, gs_regress_intercept, 34, 1, solutes)
    PUMAS3_chi_df_1 = convert_efret_df_to_chi_df('PUMA S3', PUMAS3_efret_1, gs_regress_slope, gs_regress_intercept, 34, 1, solutes)
    Ash1_chi_df_1 = convert_efret_df_to_chi_df('Ash1', Ash1_efret_1, gs_regress_slope, gs_regress_intercept, 83, 1, solutes)
    E1A_chi_df_1 = convert_efret_df_to_chi_df('E1A', E1A_efret_1, gs_regress_slope, gs_regress_intercept, 40, 1, solutes)
    P53_chi_df_1 = convert_efret_df_to_chi_df('p53', P53_efret_1, gs_regress_slope, gs_regress_intercept, 61, 1, solutes)
    FUS_chi_df_1 = convert_efret_df_to_chi_df('FUS', FUS_efret_1, gs_regress_slope, gs_regress_intercept, 163, 1, solutes)
    GS0_chi_df_2 = convert_efret_df_to_chi_df('GS0', GS0_efret_2, gs_regress_slope, gs_regress_intercept, 0, 2, solutes)
    GS8_chi_df_2 = convert_efret_df_to_chi_df('GS8', GS8_efret_2, gs_regress_slope, gs_regress_intercept, 16, 2, solutes)
    GS16_chi_df_2 = convert_efret_df_to_chi_df('GS16', GS16_efret_2, gs_regress_slope, gs_regress_intercept, 32, 2, solutes)
    GS24_chi_df_2 = convert_efret_df_to_chi_df('GS24', GS24_efret_2, gs_regress_slope, gs_regress_intercept, 48, 2, solutes)
    GS32_chi_df_2 = convert_efret_df_to_chi_df('GS32', GS32_efret_2, gs_regress_slope, gs_regress_intercept, 64, 2, solutes)
    GS48_chi_df_2 = convert_efret_df_to_chi_df('GS48', GS48_efret_2, gs_regress_slope, gs_regress_intercept, 96, 2, solutes)
    PUMA_chi_df_2 = convert_efret_df_to_chi_df('PUMA WT', PUMA_efret_2, gs_regress_slope, gs_regress_intercept, 34, 2, solutes)
    PUMAS1_chi_df_2 = convert_efret_df_to_chi_df('PUMA S1', PUMAS1_efret_2, gs_regress_slope, gs_regress_intercept, 34, 2, solutes)
    PUMAS2_chi_df_2 = convert_efret_df_to_chi_df('PUMA S2', PUMAS2_efret_2, gs_regress_slope, gs_regress_intercept, 34, 2, solutes)
    PUMAS3_chi_df_2 = convert_efret_df_to_chi_df('PUMA S3', PUMAS3_efret_2, gs_regress_slope, gs_regress_intercept, 34, 2, solutes)
    Ash1_chi_df_2 = convert_efret_df_to_chi_df('Ash1', Ash1_efret_2, gs_regress_slope, gs_regress_intercept, 83, 2, solutes)
    E1A_chi_df_2 = convert_efret_df_to_chi_df('E1A', E1A_efret_2, gs_regress_slope, gs_regress_intercept, 40, 2, solutes)
    P53_chi_df_2 = convert_efret_df_to_chi_df('p53', P53_efret_2, gs_regress_slope, gs_regress_intercept, 61, 2, solutes)
    FUS_chi_df_2 = convert_efret_df_to_chi_df('FUS', FUS_efret_2, gs_regress_slope, gs_regress_intercept, 163, 2, solutes)
    
    print ('Chi values computed!')
    
    # Combine chi values for all IDRs into one data frame for use in building figures
    all_idrs_chi_df = pd.concat([GS0_chi_df_1,GS8_chi_df_1,GS16_chi_df_1,GS24_chi_df_1,GS32_chi_df_1,GS48_chi_df_1,
                                 PUMA_chi_df_1,PUMAS1_chi_df_1,PUMAS2_chi_df_1,PUMAS3_chi_df_1,
                                 Ash1_chi_df_1,E1A_chi_df_1,P53_chi_df_1,FUS_chi_df_1,
                                 GS0_chi_df_2,GS8_chi_df_2,GS16_chi_df_2,GS24_chi_df_2,GS32_chi_df_2,GS48_chi_df_2,
                                 PUMA_chi_df_2,PUMAS1_chi_df_2,PUMAS2_chi_df_2,PUMAS3_chi_df_2,
                                 Ash1_chi_df_2,E1A_chi_df_2,P53_chi_df_2,FUS_chi_df_2], ignore_index=True)
    
    all_idrs_chi_df.to_csv('all_idrs_chi_df.csv', index=False)                             
    return (all_idrs_chi_df)
    
# get_all_idrs_chi_df() 