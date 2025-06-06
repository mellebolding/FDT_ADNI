
import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, detrend
from functions_FDT_numba_v8 import *
from functions_FC_v3 import *
from functions_LinHopf_Ceff_sigma_fit_v3 import LinHopf_Ceff_sigma_fitting_numba

import matplotlib.pyplot as plt
import matplotlib.ticker as tck
from scipy.linalg import expm
import pandas as pd
import scipy.integrate as integrate
from scipy.linalg import solve_continuous_lyapunov
from DataLoaders.baseDataLoader import DataLoader
import ADNI_A
from functions_FDT_numba_v8 import *
from functions_boxplots_WN3_v0 import *
from functions_violinplots_WN3_v0 import *
from functions_FDT_numba_v8 import construct_matrix_A, Integrate_Langevin_ND_Optimized, closest_valid_M
import filterps
import functions_boxplots_WN3_v0
from typing import Union
from numba import njit, prange, objmode
import time

 ### Append variables to npz file
import os
def append_to_npz(filename, **new_data):
    """
    Appends new variables to an existing .npz file or creates one if it doesn't exist.

    Parameters:
    - filename (str): Path to the .npz file.
    - new_data (dict): Keyword arguments representing variables to add.
    """
    if os.path.exists(filename):
        # Load existing data
        existing_data = dict(np.load(filename))
    else:
        existing_data = {}

    # Update with new variables
    existing_data.update(new_data)

    # Save back to file
    np.savez(filename, **existing_data)

### z-scoring data
from scipy.signal import detrend as scipy_detrend
def zscore_time_series(data, mode='parcel', detrend=False):
    """
    Optionally detrend and z-score the time series either parcel-wise or globally.

    Parameters:
    - data: numpy array of shape [NSUB, NPARCELS, NTIMES] or [NPARCELS, NTIMES]
    - mode: str, either 'parcel' (default) or 'global'
        - 'parcel': z-score each parcel individually across time
        - 'global': z-score the entire time series (per subject if 3D)
        - 'none': leave unchanged
    - detrend: bool, whether to remove linear trend along the time axis

    Returns:
    - processed_data: numpy array of the same shape, detrended and/or z-scored
    """
    if mode not in ['parcel', 'global', 'none']:
        raise ValueError("mode must be 'parcel', 'global' or 'none'")

    data = np.asarray(data)

    # Apply detrending if requested
    if detrend:
        if data.ndim == 2:
            # [NPARCELS, NTIMES]
            data = scipy_detrend(data, axis=1, type='linear')
        elif data.ndim == 3:
            # [NSUB, NPARCELS, NTIMES]
            # Detrend along the time axis (axis=2) for each parcel of each subject
            data = scipy_detrend(data, axis=2, type='linear')
        else:
            raise ValueError("Input data must be 2D or 3D for detrending.")

    # Apply z-scoring
    if mode == 'none':
        return data
    if data.ndim == 2:
        if mode == 'parcel':
            mean = np.mean(data, axis=1, keepdims=True)
            std = np.std(data, axis=1, keepdims=True)
        elif mode == 'global':
            mean = np.mean(data, keepdims=True)
            std = np.std(data, keepdims=True)
    elif data.ndim == 3:
        if mode == 'parcel':
            mean = np.mean(data, axis=2, keepdims=True)
            std = np.std(data, axis=2, keepdims=True)
        elif mode == 'global':
            mean = np.mean(data, axis=(1, 2), keepdims=True)
            std = np.std(data, axis=(1, 2), keepdims=True)
    else:
        raise ValueError("Input data must be 2D or 3D.")

    std[std == 0] = 1.0  # avoid division by zero
    return (data - mean) / std

### Filter data
import numpy as np
from scipy.signal import filtfilt, detrend as scipy_detrend
def filter_time_series(data, bfilt, afilt, detrend=True):
    """
    Optionally detrend and filter time series using filtfilt.

    Parameters:
    - data: np.ndarray of shape [NSUB, NPARCELS, NTIMES] or [NPARCELS, NTIMES]
    - bfilt, afilt: filter coefficients for filtfilt
    - detrend: bool, whether to detrend each time series before filtering

    Returns:
    - filtered_data: np.ndarray of the same shape, filtered time series
    """
    data = np.asarray(data)
    if data.ndim == 2:
        # Single subject case: [NPARCELS, NTIMES]
        NPARCELS, NTIMES = data.shape
        filtered_data = np.zeros_like(data)
        for parcel in range(NPARCELS):
            ts = data[parcel, :]
            if detrend:
                ts = scipy_detrend(ts, type='linear')
            ts = ts - np.mean(ts)
            filtered_data[parcel, :] = filtfilt(bfilt, afilt, ts)
    elif data.ndim == 3:
        # Group case: [NSUB, NPARCELS, NTIMES]
        NSUB, NPARCELS, NTIMES = data.shape
        filtered_data = np.zeros_like(data)
        for sub in range(NSUB):
            for parcel in range(NPARCELS):
                ts = data[sub, parcel, :]
                if detrend:
                    ts = scipy_detrend(ts, type='linear')
                ts = ts - np.mean(ts)
                filtered_data[sub, parcel, :] = filtfilt(bfilt, afilt, ts)
    else:
        raise ValueError("Input data must be 2D or 3D.")
    return filtered_data
def calc_H_freq(
        all_HC_fMRI: Union[np.ndarray, dict], 
        tr: float, 
        version: filterps.FiltPowSpetraVersion=filterps.FiltPowSpetraVersion.v2021
    ):
        """
        Compute H freq for each node. 
        
        Parameters
        ----------
        all_HC_fMRI: The fMRI of the "health control" group. Can be given in a dictionaray format, 
                     or in an array format (subject, time, node).
                     NOTE: that the signals must already be filitered. 
        tr: TR in milliseconds
        version: Version of FiltPowSpectra to use

        Returns
        -------
        The h frequencies for each node
        """
        f_diff = filterps.filt_pow_spetra_multiple_subjects(all_HC_fMRI, tr, version)
        return f_diff 


### ADNI Loading the data and data loader
DL = ADNI_A.ADNI_A()

# example of individual
sc = DL.get_subjectData('002_s_0413')
SC = sc['002_s_0413']['SC'] # Structural connectivity

# Loading the data for all subjects
HC_IDs = DL.get_groupSubjects('HC')
HC_MRI = {}
for subject in HC_IDs:
    data = DL.get_subjectData(subject)
    HC_MRI[subject] = data[subject]['timeseries']

MCI_IDs = DL.get_groupSubjects('MCI')
MCI_MRI = {}
for subject in MCI_IDs:
    data = DL.get_subjectData(subject)
    MCI_MRI[subject] = data[subject]['timeseries']

AD_IDs = DL.get_groupSubjects('AD')
AD_MRI = {}
for subject in AD_IDs:
    data = DL.get_subjectData(subject)
    AD_MRI[subject] = data[subject]['timeseries']

# Okay this is loading in the effecetive connectivity, so we cannot use this for f_diff
# we need to use the data loader to get the timeseries data
#EC_HC_data = scipy.io.loadmat('ADNI-A_DATA/EC_filterted/HC_FDT_results_filters0109.mat')
#EC_MCI_data = scipy.io.loadmat('ADNI-A_DATA/EC_filterted/MCI_FDT_results_filters0109.mat')
#EC_AD_data = scipy.io.loadmat('ADNI-A_DATA/EC_filterted/AD_FDT_results_filters0109.mat')
#print(EC_HC_data.keys()) # check the keys

### Set conditions
NPARCELLS = 15 #tot: 379
Tau = 1
TR = 2
a_param = -0.02
min_sigma_val = 1e-7
gconst = 1.0
avec = a_param * np.ones(NPARCELLS)
Ndim = 2 * NPARCELLS
v0bias = 0.0
t0 = 0
tfinal = 200
dt = 0.01
times = np.arange(t0, tfinal+dt, dt)

group_names = ['HC', 'MCI', 'AD']
group_sizes = {'HC': len(HC_IDs), 'MCI': len(MCI_IDs), 'AD': len(AD_IDs)}
cond_index_map = {'HC': 0, 'MCI': 1, 'AD': 2}
I_FDT_all = np.full((3, NPARCELLS), np.nan)
#I_FDT_all = np.full((3, max(group_sizes.values()), NPARCELLS), np.nan)


for i in range(1,4):
    COND = i
    if COND == 1: ## --> HC
        #Ceffgroup = EC_HC_data['Ceff_subjects']
        f_diff = calc_H_freq(HC_MRI, 3000, filterps.FiltPowSpetraVersion.v2021)
        ts_gr = HC_MRI
        ID = HC_IDs

    elif COND == 2: ## --> MCI
        #Ceffgroup = EC_MCI_data['Ceff_subjects']
        f_diff = calc_H_freq(MCI_MRI, 3000, filterps.FiltPowSpetraVersion.v2021)
        ts_gr = MCI_MRI
        ID = MCI_IDs

    elif COND == 3: ## --> AD
        #Ceffgroup = EC_AD_data['Ceff_subjects']
        f_diff = calc_H_freq(AD_MRI, 3000, filterps.FiltPowSpetraVersion.v2021)
        ts_gr = AD_MRI
        ID = AD_IDs

    f_diff = f_diff[:NPARCELLS] # frequencies of group
    omega = 2 * np.pi * f_diff

    ### Generates a "group" TS with the same length for all subjects
    min_ntimes = min(ts_gr[subj_id].shape[1] for subj_id in ID)
    ts_gr_arr = np.zeros((len(ID), NPARCELLS, min_ntimes))
    for sub in range(len(ID)):
        subj_id = ID[sub]
        ts_gr_arr[sub,:,:] = ts_gr[subj_id][:NPARCELLS, :min_ntimes].copy() 
    TSemp_zsc = zscore_time_series(ts_gr_arr, mode='global', detrend=True)[:,:NPARCELLS,:].copy() #mode: parcel, global, none
    TSemp_fit_group = np.zeros((len(ID), NPARCELLS, min_ntimes))
    TSemp_fit_group = TSemp_zsc[:,:NPARCELLS, :].copy()

    SC_N = SC[:NPARCELLS, :NPARCELLS]
    SC_N /= np.max(SC_N)
    SC_N *= 0.2
    Ceff_ini = SC_N.copy()
    sigma_mean = 0.45
    sigma_ini = sigma_mean * np.ones(NPARCELLS)
        
    start_time = time.time()
    Ceff_group, sigma_group, FCemp_group, FCsim_group, error_iter_group = \
                                LinHopf_Ceff_sigma_fitting_numba(TSemp_fit_group, Ceff_ini, NPARCELLS, TR, f_diff, sigma_ini, Tau=1,
                                epsFC_Ceff=7e-5, epsCOVtau_Ceff=1e-5, epsFC_sigma=7e-5, epsCOVtau_sigma=1e-5,
                                MAXiter=10000, error_tol=1e-6, patience=5, learning_rate_factor=0.8,
                                Ceff_norm=False, maxC=0.2)
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.4f} seconds")

    plt.plot(np.arange(1,len(error_iter_group)+1)*100,error_iter_group, 'o-', label='error @100 iter')
    plt.xlabel('iter')
    plt.legend()

    fig_name = f"FCmatrices_group_{group_names[COND - 1]}.png"
    save_path = f"/home/mbolding/figures/{fig_name}"
    plot_FC_matrices(FCemp_group, FCsim_group, title1="group FCemp", title2="group FCsim", save_path=save_path, size=1, dpi=300)
    fig_name = f"ECmatrix_group_{group_names[COND - 1]}.png"
    save_path = f"/home/mbolding/figures/{fig_name}"
    plot_FC_matrix(Ceff_group, title="group Ceff fitted", size=1.1, save_path=save_path,dpi=300)
    
     ## Plot sigma
    plt.figure(figsize=(np.clip(NPARCELLS, 8, 12), 4))
    plt.plot(range(1, NPARCELLS+1), sigma_ini, '.--', color='gray', alpha=0.5, label='Initial guess')
    plt.plot(range(1, NPARCELLS+1), sigma_group, '.-', color='tab:blue', alpha=1, label='sigma fit normalized')
    plt.axhline(np.mean(sigma_group), color='tab:blue', linestyle='--', label=f'{np.mean(sigma_group):.5f}')
    plt.xlabel('Parcels')
    ticks = np.arange(1, NPARCELLS + 1)
    labels = [str(ticks[0])] + [''] * (len(ticks) - 2) + [str(ticks[-1])]
    plt.xticks(ticks,labels)
    plt.legend()
    plt.show()
    sigma_group_2 = np.append(sigma_group, sigma_group)
    v0std = sigma_group_2
    print(sigma_group)

    
    Gamma = -construct_matrix_A(avec, omega, Ceff_group, gconst)

    v0 = v0std * np.random.standard_normal(Ndim) + v0bias
    vsim, noise = Integrate_Langevin_ND_Optimized(Gamma, sigma_group_2, initcond=v0, duration=tfinal, integstep=dt)

    v0 = vsim[:,-1]
    vsim, noise = Integrate_Langevin_ND_Optimized(Gamma, sigma_group_2, initcond=v0, duration=tfinal, integstep=dt)
        
    D = np.diag(sigma_group_2**2 * np.ones(Ndim))
    V_0 = solve_continuous_lyapunov(Gamma, D)

    tmax = 100
    ts0 = 0
    I_tmax_s0 = Its_Langevin_ND(Gamma, sigma_group_2, V_0, tmax, ts0)[0:NPARCELLS]

    group_name = group_names[COND - 1]
    group_idx = cond_index_map[group_name]
    #subject_idx = sub  # Already incrementing

    I_FDT_all[group_idx, :] = I_tmax_s0

I_FDT_all_sub = np.full((3, max(group_sizes.values()), NPARCELLS), np.nan)


for i in range(1,4):
    COND = i
    if COND == 1: ## --> HC
        #Ceffgroup = EC_HC_data['Ceff_subjects']
        f_diff = calc_H_freq(HC_MRI, 3000, filterps.FiltPowSpetraVersion.v2021)
        ts_gr = HC_MRI
        ID = HC_IDs

    elif COND == 2: ## --> MCI
        #Ceffgroup = EC_MCI_data['Ceff_subjects']
        f_diff = calc_H_freq(MCI_MRI, 3000, filterps.FiltPowSpetraVersion.v2021)
        ts_gr = MCI_MRI
        ID = MCI_IDs

    elif COND == 3: ## --> AD
        #Ceffgroup = EC_AD_data['Ceff_subjects']
        f_diff = calc_H_freq(AD_MRI, 3000, filterps.FiltPowSpetraVersion.v2021)
        ts_gr = AD_MRI
        ID = AD_IDs

    f_diff = f_diff[:NPARCELLS] # frequencies of group
    omega = 2 * np.pi * f_diff

    ### Generates a "group" TS with the same length for all subjects
    min_ntimes = min(ts_gr[subj_id].shape[1] for subj_id in ID)
    ts_gr_arr = np.zeros((len(ID), NPARCELLS, min_ntimes))
    for sub in range(len(ID)):
        subj_id = ID[sub]
        ts_gr_arr[sub,:,:] = ts_gr[subj_id][:NPARCELLS, :min_ntimes].copy() 
    TSemp_zsc = zscore_time_series(ts_gr_arr, mode='global', detrend=True)[:,:NPARCELLS,:].copy() #mode: parcel, global, none
    TSemp_fit_group = np.zeros((len(ID), NPARCELLS, min_ntimes))
    TSemp_fit_group = TSemp_zsc[:,:NPARCELLS, :].copy()

    SC_N = SC[:NPARCELLS, :NPARCELLS]
    SC_N /= np.max(SC_N)
    SC_N *= 0.2
    Ceff_ini = SC_N.copy()
    sigma_mean = 0.45
    sigma_ini = sigma_mean * np.ones(NPARCELLS)

    Ceff_sub = np.zeros((len(ID), NPARCELLS, NPARCELLS))
    sigma_sub = np.zeros((len(ID), NPARCELLS))
    FCemp_sub = np.zeros((len(ID), NPARCELLS, NPARCELLS))
    FCsim_sub = np.zeros((len(ID), NPARCELLS, NPARCELLS))
    error_iter_sub = np.ones((len(ID), 200)) * np.nan

    for sub in range(len(ID)):
        subj_id = ID[sub]
        #Ceff = Ceffgroup[sub][:NPARCELLS,:NPARCELLS] # effecitve connectivity
        f_diff = f_diff[:NPARCELLS] # frequencies of group
        omega = 2 * np.pi * f_diff

        TSemp_fit_sub = TSemp_zsc[sub, :, :].copy()  # time series for the subject
        
        Ceff_sub[sub], sigma_sub[sub], FCemp_sub[sub], FCsim_sub[sub], error_iter_sub_aux = \
                                    LinHopf_Ceff_sigma_fitting_numba(TSemp_fit_sub, Ceff_group, NPARCELLS, TR, f_diff, sigma_group, Tau=1,
                                    epsFC_Ceff=7e-5, epsCOVtau_Ceff=1e-5, epsFC_sigma=7e-5, epsCOVtau_sigma=1e-5,
                                    MAXiter=10000, error_tol=1e-4, patience=5, learning_rate_factor=0.8,
                                    Ceff_norm=False, maxC=0.2)
        error_iter_sub[sub, :len(error_iter_sub_aux)] = error_iter_sub_aux

        sigma_vec = np.append(sigma_sub[sub], sigma_sub[sub]).copy()  # double the sigma for the x and y components
        v0std = sigma_vec[sub] 

    
        Gamma = (-1) * construct_matrix_A(avec, omega, Ceff_sub[sub], gconst)
        term_time = 100 * TR
        v0 = v0std * np.random.standard_normal(Ndim) + v0bias
        vsim, noise = Integrate_Langevin_ND_Optimized(Gamma, sigma_vec, initcond=v0, duration=term_time, integstep=dt)

        v0 = vsim[:,-1]
        vsim, noise = Integrate_Langevin_ND_Optimized(Gamma, sigma_vec, initcond=v0, duration=tfinal, integstep=dt)
        
        D = np.diag(sigma_vec**2 * np.ones(Ndim))
        V_0 = solve_continuous_lyapunov(Gamma, D)

        tmax = 100
        ts0 = 0
        I_tmax_s0 = Its_Langevin_ND(Gamma, sigma_vec, V_0, tmax, ts0)[0:NPARCELLS]

        group_name = group_names[COND - 1]
        group_idx = cond_index_map[group_name]

        I_FDT_all_sub[group_idx, sub, :] = I_tmax_s0
        
    
        plt.plot(np.arange(1,len(error_iter_sub[sub])+1)*100,error_iter_sub[sub], 'o-', label='error @100 iter')
        plt.xlabel('iter')
        plt.legend()

        plot_FC_matrices(FCemp_sub[sub], FCsim_sub[sub], title1=f"FCemp sub.{sub+1}", title2=f"FCsim sub.{sub+1}", size=1, dpi=300)

        plot_FC_matrix(Ceff_sub[sub], title=f"Ceff fitted sub. {sub+1}", size=1.1, dpi=300)
    
     ## Plot sigma
        plt.figure(figsize=(np.clip(NPARCELLS, 8, 12), 4))
        plt.plot(range(1, NPARCELLS+1), sigma_group, '.-', color='tab:blue', alpha=1, lw=2, label='sigma fit normalized (group)')
        plt.plot(range(1, NPARCELLS+1), sigma_sub[sub], '.-', color='tab:red', alpha=1, label=f'sigma fit normalized (sub. {sub+1})')
        plt.axhline(np.mean(sigma_sub[sub]), color='tab:blue', linestyle='--', label=f'{np.mean(sigma_sub[sub]):.5f}')
        plt.xlabel('Parcels')
        ticks = np.arange(1, NPARCELLS + 1)
        labels = [str(ticks[0])] + [''] * (len(ticks) - 2) + [str(ticks[-1])]
        plt.xticks(ticks,labels)
        plt.legend()
        plt.show()



