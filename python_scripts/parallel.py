# parallel and restricted version of EC_sigma_fit.py (using concurrent.futures)
# Idea: one file to get both parcel and subject results (i.e. Ceff and sigma, optionally), the subects will be parallelized
import os
import sys
# Absolute :path to the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Absolute path to the repo root (one level up from this script)
repo_root = os.path.abspath(os.path.join(script_dir, '..'))

os.chdir(repo_root)

sys.path.insert(0, repo_root)
sys.path.insert(0, os.path.join(repo_root, 'support_files'))
sys.path.insert(0, os.path.join(repo_root, 'DataLoaders'))


base_folder = os.path.join(repo_root, 'ADNI-A_DATA')
connectome_dir = os.path.join(base_folder, 'connectomes')
results_dir = os.path.join(repo_root, 'Result_plots')
ECgroup_subfolder = os.path.join(results_dir, 'EC_group')
Ceff_sigma_subfolder = os.path.join(results_dir, 'Ceff_sigma_results')
ECsub_subfolder = os.path.join(results_dir, 'EC_sub')
FCgroup_subfolder = os.path.join(results_dir, 'FC_group')
FCsub_subfolder = os.path.join(results_dir, 'FC_sub')
sigma_subfolder = os.path.join(results_dir, 'sig_sub')
sigma_group_subfolder = os.path.join(results_dir, 'sig_group')
FDT_parcel_subfolder = os.path.join(results_dir, 'FDT_parcel')
FDT_subject_subfolder = os.path.join(results_dir, 'FDT_sub')
Inorm1_group_subfolder = os.path.join(results_dir, 'Inorm1_group')
Inorm2_group_subfolder = os.path.join(results_dir, 'Inorm2_group')
Inorm1_sub_subfolder = os.path.join(results_dir, 'Inorm1_sub')
Inorm2_sub_subfolder = os.path.join(results_dir, 'Inorm2_sub')
training_dir = os.path.join(results_dir, 'training_conv')
os.makedirs(results_dir, exist_ok=True)
os.makedirs(ECgroup_subfolder, exist_ok=True)
os.makedirs(ECsub_subfolder, exist_ok=True)
os.makedirs(FCgroup_subfolder, exist_ok=True)
os.makedirs(Ceff_sigma_subfolder, exist_ok=True)
os.makedirs(FCsub_subfolder, exist_ok=True)
os.makedirs(sigma_subfolder, exist_ok=True)
os.makedirs(sigma_group_subfolder, exist_ok=True)
os.makedirs(FDT_parcel_subfolder, exist_ok=True)
os.makedirs(FDT_subject_subfolder, exist_ok=True)
os.makedirs(Inorm1_group_subfolder, exist_ok=True)
os.makedirs(Inorm2_group_subfolder, exist_ok=True)
os.makedirs(Inorm1_sub_subfolder, exist_ok=True)
os.makedirs(Inorm2_sub_subfolder, exist_ok=True)
os.makedirs(training_dir, exist_ok=True)

import os
import sys
from functions_FDT_numba_v9 import *
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.signal import detrend as scipy_detrend
from functions_FC_v3 import *
from functions_LinHopf_Ceff_sigma_fit_v6 import LinHopf_Ceff_sigma_fitting_numba
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
from functions_violinplots_v2 import *
from functions_FDT_numba_v8 import construct_matrix_A, Integrate_Langevin_ND_Optimized, closest_valid_M
import filterps
import functions_boxplots_WN3_v0
from typing import Union
from numba import njit, prange, objmode
import time
import p_values as p_values
import statannotations_permutation

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

def clear_npz_file(folder, filename):
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)
    # Save an empty records array, overwriting any existing file
    np.savez(filepath, records=np.array([], dtype=object))


def append_record_to_npz(folder, filename, **record):
    """
    Appends a record (dict) to a 'records' array in a .npz file located in `folder`.
    Creates the folder and file if they don't exist.

    Parameters
    ----------
    folder : str
        Path to the subfolder where the file will be saved.
    filename : str
        Name of the .npz file (e.g., 'Ceff_sigma_results.npz').
    record : dict
        Arbitrary key-value pairs to store (arrays, strings, numbers, etc.).
    """
    os.makedirs(folder, exist_ok=True)  # ensure subfolder exists
    filepath = os.path.join(folder, filename)

    if os.path.exists(filepath):
        existing_data = dict(np.load(filepath, allow_pickle=True))
        records = list(existing_data.get("records", []))
    else:
        records = []

    records.append(record)
    np.savez(filepath, records=np.array(records, dtype=object))


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


###### Loading the data ######
DL = ADNI_A.ADNI_A()

# example of individual
subdata = DL.get_subjectData('002_S_0413')
SC = subdata['002_S_0413']['SC'] # Structural connectivity

# Loading the timeseries data for all subjects and dividing them into groups
HC_IDs = DL.get_groupSubjects('HC')
HC_MRI = {}
for subject in HC_IDs:
    data = DL.get_subjectData(subject,printInfo=False)
    HC_MRI[subject] = data[subject]['timeseries'].T

MCI_IDs = DL.get_groupSubjects('MCI')
MCI_MRI = {}
for subject in MCI_IDs:
    data = DL.get_subjectData(subject,printInfo=False)
    MCI_MRI[subject] = data[subject]['timeseries'].T

AD_IDs = DL.get_groupSubjects('AD')
AD_MRI = {}
for subject in AD_IDs:
    data = DL.get_subjectData(subject,printInfo=False)
    AD_MRI[subject] = data[subject]['timeseries'].T


### Set conditions
NPARCELLS = 18 #tot: 379
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
sigma_mean = 0.45
CEFF_FITTING = True
SIGMA_FITTING = False
if SIGMA_FITTING: NOISE_TYPE = 'hetero'
else: NOISE_TYPE = 'homo'
COMPETITIVE_COUPLING = False
CEFF_NORMALIZATION = True
maxC = 0.2
iter_check_group = 100
fit_Ceff=CEFF_FITTING
competitive_coupling=COMPETITIVE_COUPLING
epsFC_Ceff = 4e-4
epsCOVtau_Ceff = 1e-4

fit_sigma=SIGMA_FITTING
sigma_reset=False
epsFC_sigma = 8e-5
epsCOVtau_sigma = 3e-5
    
MAXiter = 10000
error_tol = 1e-3
patience = 5
learning_rate_factor = 1.0
Ceff_norm=CEFF_NORMALIZATION
maxC=maxC
iter_check=iter_check_group

group_names = ['HC', 'MCI', 'AD']
group_sizes = {'HC': len(HC_IDs), 'MCI': len(MCI_IDs), 'AD': len(AD_IDs)}
cond_index_map = {'HC': 0, 'MCI': 1, 'AD': 2}
I_FDT_all = np.full((3, NPARCELLS), np.nan)
Inorm1_tmax_s0_group = np.zeros((3, NPARCELLS))
Inorm2_tmax_s0_group = np.zeros((3, NPARCELLS))

clear_npz_file(Ceff_sigma_subfolder, f"Ceff_sigma_{NPARCELLS}_{NOISE_TYPE}.npz")

### Group level
for COND in range(1,4):
    if COND == 1: ## --> HC
        f_diff = calc_H_freq(HC_MRI, 3000, filterps.FiltPowSpetraVersion.v2021)[0]
        ts_gr = HC_MRI
        ID = HC_IDs

    elif COND == 2: ## --> MCI
        f_diff = calc_H_freq(MCI_MRI, 3000, filterps.FiltPowSpetraVersion.v2021)[0]
        ts_gr = MCI_MRI
        ID = MCI_IDs

    elif COND == 3: ## --> AD
        f_diff = calc_H_freq(AD_MRI, 3000, filterps.FiltPowSpetraVersion.v2021)[0]
        ts_gr = AD_MRI
        ID = AD_IDs
    
    f_diff = f_diff[:NPARCELLS] # frequencies of group
    omega = 2 * np.pi * f_diff

    ### Generates a "group" TS with the same length for all subjects
    min_ntimes = min(ts_gr[subj_id].shape[0] for subj_id in ID)
    ts_gr_arr = np.zeros((len(ID), NPARCELLS, min_ntimes))
    for sub in range(len(ID)):
        subj_id = ID[sub]
        ts_gr_arr[sub,:,:] = ts_gr[subj_id][:min_ntimes,:NPARCELLS].T.copy() 
    TSemp_zsc = zscore_time_series(ts_gr_arr, mode='global', detrend=True)[:,:NPARCELLS,:].copy() #mode: parcel, global, none
    TSemp_fit_group = np.zeros((len(ID), NPARCELLS, min_ntimes))
    TSemp_fit_group = TSemp_zsc[:,:NPARCELLS, :].copy()

    SC_N = SC[:NPARCELLS, :NPARCELLS]
    SC_N /= np.max(SC_N)
    SC_N *= 0.2
    Ceff_ini = SC_N.copy()
    sigma_ini = sigma_mean * np.ones(NPARCELLS)

    

    start_time = time.time()
    Ceff_group, sigma_group, FCemp_group, FCsim_group, error_iter_group, errorFC_iter_group, errorCOVtau_iter_group, = \
                                LinHopf_Ceff_sigma_fitting_numba(TSemp_fit_group, Ceff_ini, NPARCELLS, TR, f_diff, sigma_ini, Tau=Tau,
                                fit_Ceff=fit_Ceff, competitive_coupling=competitive_coupling, 
                                fit_sigma=False, sigma_reset=sigma_reset,
                                epsFC_Ceff=epsFC_Ceff, epsCOVtau_Ceff=epsCOVtau_Ceff, epsFC_sigma=epsFC_sigma, epsCOVtau_sigma=epsCOVtau_sigma,
                                MAXiter=MAXiter, error_tol=error_tol, patience=patience, learning_rate_factor=learning_rate_factor,
                                Ceff_norm=Ceff_norm, maxC=maxC,
                                iter_check=iter_check, plot_evol=False, plot_evol_last=False)
    end_time = time.time()
    print(f"Execution time group: {end_time - start_time:.4f} seconds")

    ## ploting the error iter
    figure_name = f"error_iter_N{NPARCELLS}_group_{group_names[COND - 1]}_{NOISE_TYPE}.png"
    save_path = os.path.join(training_dir, figure_name)
    plt.figure()
    plt.plot(np.arange(1, len(error_iter_group) + 1) * 100, error_iter_group, 'o-', label='error @100 iter')
    plt.xlabel('iter')
    plt.legend()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    ## plotting the FC and Ceff matrices
    fig_name = f"FCmatrices_N{NPARCELLS}_group_{group_names[COND - 1]}_{NOISE_TYPE}.png"
    save_path = os.path.join(FCgroup_subfolder, fig_name)
    plot_FC_matrices(FCemp_group, FCsim_group, title1="group FCemp", title2="group FCsim", save_path=save_path, size=1, dpi=300)
    fig_name = f"ECmatrix_N{NPARCELLS}_group_{group_names[COND - 1]}_{NOISE_TYPE}.png"
    save_path = os.path.join(ECgroup_subfolder, fig_name)
    plot_FC_matrix(Ceff_group, title="group Ceff fitted", size=1.1, save_path=save_path,dpi=300)

    ## plot the sigma
    fig_name = f"sigma_fit_N_{NPARCELLS}_group_{group_names[COND - 1]}_{NOISE_TYPE}.png"
    save_path = os.path.join(sigma_group_subfolder, fig_name)
    plt.figure(figsize=(np.clip(NPARCELLS, 8, 12), 4))
    plt.plot(range(1, NPARCELLS+1), sigma_ini, '.--', color='gray', alpha=0.5, label='Initial guess')
    plt.plot(range(1, NPARCELLS+1), sigma_group, '.-', color='tab:blue', alpha=1, label='sigma fit normalized')
    plt.axhline(np.mean(sigma_group), color='tab:blue', linestyle='--', label=f'{np.mean(sigma_group):.5f}')
    plt.xlabel('Parcels')
    ticks = np.arange(1, NPARCELLS + 1)
    labels = [str(ticks[0])] + [''] * (len(ticks) - 2) + [str(ticks[-1])]
    plt.xticks(ticks, labels)
    plt.legend()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    ## save the results
    
    append_record_to_npz(
    Ceff_sigma_subfolder,
    f"Ceff_sigma_{NPARCELLS}_{NOISE_TYPE}.npz",
    level="group",
    condition=f"{COND}",
    sigma=sigma_group,
    Ceff=Ceff_group,
    omega=omega)


### subject level
# Note: The subject level calculations are done in parallel for each condition
for i in range(1,4):
    COND = i
    if COND == 1: ## --> HC
        #f_diff = calc_H_freq(HC_MRI, 3000, filterps.FiltPowSpetraVersion.v2021)
        ts_gr = HC_MRI
        ID = HC_IDs

    elif COND == 2: ## --> MCI
        #f_diff = calc_H_freq(MCI_MRI, 3000, filterps.FiltPowSpetraVersion.v2021)
        ts_gr = MCI_MRI
        ID = MCI_IDs

    elif COND == 3: ## --> AD
        #f_diff = calc_H_freq(AD_MRI, 3000, filterps.FiltPowSpetraVersion.v2021)
        ts_gr = AD_MRI
        ID = AD_IDs

    #f_diff = f_diff[:NPARCELLS] # frequencies of group
    #omega = 2 * np.pi * f_diff

    ### Generates a "group" TS with the same length for all subjects
    min_ntimes = min(ts_gr[subj_id].shape[1] for subj_id in ID)
    ts_gr_arr = np.zeros((len(ID), NPARCELLS, min_ntimes))
    for sub in range(len(ID)):
        subj_id = ID[sub]
        ts_gr_arr[sub,:,:] = ts_gr[subj_id][:NPARCELLS, :min_ntimes].copy() 
    TSemp_zsc = zscore_time_series(ts_gr_arr, mode='global', detrend=True)[:,:NPARCELLS,:].copy() #mode: parcel, global, none
    TSemp_fit_group = np.zeros((len(ID), NPARCELLS, min_ntimes))
    TSemp_fit_group = TSemp_zsc[:,:NPARCELLS, :].copy()

    sigma_ini = sigma_mean * np.ones(NPARCELLS)
    Ceff_sub = np.zeros((len(ID), NPARCELLS, NPARCELLS))
    sigma_sub = np.zeros((len(ID), NPARCELLS))
    FCemp_sub = np.zeros((len(ID), NPARCELLS, NPARCELLS))
    FCsim_sub = np.zeros((len(ID), NPARCELLS, NPARCELLS))
    error_iter_sub = np.ones((len(ID), 200)) * np.nan
    
    #frqs = ts_gr_arr[:,:,:].copy().T  # time series for the subject
    #print(f"frqs shape: {frqs.shape}")
    #print(f"ts_gr:", ts_gr.shape)
    f_diff = calc_H_freq(ts_gr, 3000, filterps.FiltPowSpetraVersion.v2021)[1]
    f_diff = f_diff[:,:NPARCELLS]  # frequencies of group


    for sub in range(len(ID)):
        subj_id = ID[sub]
        omega = 2 * np.pi * f_diff[sub,:]
        #print("omega shape:", omega.shape, "f_diff shape:", f_diff.shape)

        #f_diff = f_diff[:NPARCELLS] # frequencies of group
        #omega = 2 * np.pi * f_diff

        TSemp_fit_sub = TSemp_zsc[sub, :, :].copy()  # time series for the subject
        
        Ceff_sub[sub], sigma_sub[sub], FCemp_sub[sub], FCsim_sub[sub], error_iter_sub_aux, errorFC_iter_sub_aux, errorCOVtau_iter_sub_aux = \
                                            LinHopf_Ceff_sigma_fitting_numba(TSemp_fit_sub, Ceff_group, NPARCELLS, TR, f_diff[sub], sigma_group, Tau=Tau,
                                            fit_Ceff=fit_Ceff, competitive_coupling=competitive_coupling, 
                                            fit_sigma=False, sigma_reset=sigma_reset,
                                            epsFC_Ceff=epsFC_Ceff, epsCOVtau_Ceff=epsCOVtau_Ceff, epsFC_sigma=epsFC_sigma, epsCOVtau_sigma=epsCOVtau_sigma,
                                            MAXiter=MAXiter, error_tol=error_tol, patience=patience, learning_rate_factor=learning_rate_factor,
                                            Ceff_norm=Ceff_norm, maxC=maxC,
                                            iter_check=iter_check, plot_evol=False, plot_evol_last=False)
        error_iter_sub[sub, :len(error_iter_sub_aux)] = error_iter_sub_aux

        #print(f"Subject {subj_id} cond: {COND}, sigma shape: {sigma_sub[sub]}")
        figure_name = f"error_iter_N_{NPARCELLS}_group_{group_names[COND - 1]}_sub_{sub}_{NOISE_TYPE}.png"
        save_path = os.path.join(training_dir, figure_name)
        plt.figure()
        plt.plot(np.arange(1, len(error_iter_sub[sub]) + 1) * 100, error_iter_sub[sub], 'o-', label='error @100 iter')
        plt.xlabel('iter')
        plt.legend()

        # Save and close
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        fig_name = f"FCmatrices_N{NPARCELLS}_group_{group_names[COND - 1]}_sub_{sub+1}_{NOISE_TYPE}.png"
        save_path = os.path.join(FCsub_subfolder, fig_name)
        plot_FC_matrices(FCemp_sub[sub], FCsim_sub[sub], title1=f"FCemp sub.{sub+1}", title2=f"FCsim sub.{sub+1}", save_path=save_path, size=1, dpi=300)
        fig_name = f"ECmatrix_N{NPARCELLS}_group_{group_names[COND - 1]}_sub_{sub+1}_{NOISE_TYPE}.png"
        save_path = os.path.join(ECsub_subfolder, fig_name)
        plot_FC_matrix(Ceff_sub[sub], title=f"Ceff fitted sub. {sub+1}", save_path=save_path, size=1.1, dpi=300)

        fig_name = f"sigma_fit_N_{NPARCELLS}_group_{group_names[COND - 1]}_sub_{sub+1}_{NOISE_TYPE}.png"
        save_path = os.path.join(sigma_subfolder, fig_name)
        plt.figure(figsize=(np.clip(NPARCELLS, 8, 12), 4))
        plt.plot(range(1, NPARCELLS + 1), sigma_group, '.-', color='tab:blue', alpha=1, lw=2, label='sigma fit normalized (group)')
        plt.plot(range(1, NPARCELLS + 1), sigma_sub[sub], '.-', color='tab:red', alpha=1, label=f'sigma fit normalized (sub. {sub+1})')
        plt.axhline(np.mean(sigma_sub[sub]), color='tab:blue', linestyle='--', label=f'{np.mean(sigma_sub[sub]):.5f}')
        plt.xlabel('Parcels')
        ticks = np.arange(1, NPARCELLS + 1)
        labels = [str(ticks[0])] + [''] * (len(ticks) - 2) + [str(ticks[-1])]
        plt.xticks(ticks, labels)
        plt.legend()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        ## save the results
        append_record_to_npz(
        Ceff_sigma_subfolder,
        f"Ceff_sigma_{NPARCELLS}_{NOISE_TYPE}.npz",
        level="subject",
        condition=f"{COND}",
        subject=f"S{sub}",
        sigma=sigma_sub[sub],
        Ceff=Ceff_sub[sub],
        omega=omega)