import os
import sys


#### Setting up paths ####

# Absolute :path to the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Absolute path to the repo root (one level up from this script)
repo_root = os.path.abspath(os.path.join(script_dir, '..'))

os.chdir(repo_root)

sys.path.insert(0, repo_root)
sys.path.insert(0, os.path.join(repo_root, 'python_scripts'))
sys.path.insert(0, os.path.join(repo_root, 'DataLoaders'))

base_folder = os.path.join(repo_root, 'ADNI-A_DATA')
connectome_dir = os.path.join(base_folder, 'connectomes')
results_dir = os.path.join(repo_root, 'Result_plots')

from LinHopf_EC_Sig_A_fit_adam_numba import LinHopf_Ceff_sigma_a_fitting_adam
from function_LinHopf_Ceff_sigma_a_fit import LinHopf_Ceff_sigma_a_fitting_numba
import ADNI_A
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.signal import detrend as scipy_detrend
import matplotlib.ticker as tck
from scipy.linalg import expm
import pandas as pd
import scipy.integrate as integrate
from scipy.linalg import solve_continuous_lyapunov
from DataLoaders.baseDataLoader import DataLoader
import filterps
from typing import Union
from numba import njit, prange, objmode
import time
import p_values as p_values
import statannotations_permutation
from functions_FC_v3 import plot_FC_matrices, plot_FC_matrix
from function_LinHopf_Ceff_sigma_a_fit import from_PET_to_a_global

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

def predict_a(a_fitted, ABeta_all, Tau_all, coef_matrix):
    """
    Predict a' given a_fitted, ABeta, Tau, and a coefficient matrix.
    Here a_fitted, ABeta_all, and Tau_all are lists with 2D arrays of shape (n_subjects, n_parcels).
    """
    const      = coef_matrix["const"].values[None, :]
    beta_coef  = coef_matrix["ABeta"].values[None, :]
    tau_coef   = coef_matrix["Tau"].values[None, :]
    inter_coef = coef_matrix["ABeta_x_Tau"].values[None, :]
    
    scale = (1+const
            + beta_coef * ABeta_all
            + tau_coef * Tau_all
            + inter_coef * (ABeta_all * Tau_all))

    return np.vstack(a_fitted) * scale


def calc_a_values(a_list_sub, a_list_group, ABeta_burden, Tau_burden):
    """
    Fit ONE global model, then apply it to subject-level and group-level data.
    """

    # ---------- 1) Fit global model ----------
    coef_matrix, results = from_PET_to_a_global(a_list_sub, ABeta_burden, Tau_burden)

    # ---------- 2) Prepare group averages ----------
    ABeta_burden_group = np.array([np.mean(group, axis=0) for group in ABeta_burden])
    Tau_burden_group   = np.array([np.mean(group, axis=0) for group in Tau_burden])

    # ---------- 3) Predict ----------
    ABeta_burden_all = np.vstack(ABeta_burden)
    Tau_burden_all   = np.vstack(Tau_burden)

    predicted_a = predict_a(a_list_sub,ABeta_burden_all, Tau_burden_all, coef_matrix)
    predicted_a_group = predict_a(a_list_group,ABeta_burden_group, Tau_burden_group, coef_matrix)

    return {
        "predicted_a": predicted_a,
        "predicted_a_group": predicted_a_group,
        "coef_matrix": coef_matrix,
        "results": results
    }

def show_error(error_iter, error_iter_2, errorFC_iter, errorFC_iter_2,
                errorCOVtau_iter, errorCOVtau_iter_2, sigma, sigma_2, sigma_ini,
                  a, a_2, FCemp, FCsim, FC_sim_2, label):
    """
    Want to give an indication of the fitting quality?
    options to show: final error, FC fit, COVtau fit, sigma fit, a fit
    """
    
    if error_iter is not None:
        # figure_name = f"error_iter_a{A_FITTING}_N{NPARCELLS}_{label}_{group_names[COND]}_{NOISE_TYPE}.png"
        # if label == 'group': save_path = os.path.join(error_fitting_group_subfolder, figure_name)
        # else: save_path = os.path.join(error_fitting_sub_subfolder, figure_name)
        plt.figure(figsize=(8,5))
        plt.plot(np.arange(1, len(error_iter) + 1) * 100, error_iter, 'o-', color='tab:blue', label='Error @100 iter')
        plt.plot(np.arange(1, len(errorFC_iter) + 1) * 100, errorFC_iter, 's-', color='tab:orange', label='Error FC @100 iter')
        plt.plot(np.arange(1, len(errorCOVtau_iter) + 1) * 100, errorCOVtau_iter, '^-', color='tab:green', label='Error COVtau @100 iter')
        if error_iter_2 is not None:
            plt.plot(np.arange(1, len(error_iter_2) + 1) * 100, error_iter_2, 'o--', color='tab:blue', label='Error Adam @100 iter')
            plt.plot(np.arange(1, len(errorFC_iter_2) + 1) * 100, errorFC_iter_2, 's--', color='tab:orange', label='Error FC Adam @100 iter')
            plt.plot(np.arange(1, len(errorCOVtau_iter_2) + 1) * 100, errorCOVtau_iter_2, '^--', color='tab:green', label='Error COVtau Adam @100 iter')
            print('Final errors:', error_iter[-1], error_iter_2[-1])
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.title(f"Error Curves - Group {group_names[COND]}")
        plt.legend(loc='upper right', fontsize=10)
        plt.grid(True)
        plt.show()
        # plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        

    ## plotting the FC and Ceff matrices
    # fig_name = f"FCmatrices_a{A_FITTING}_N{NPARCELLS}_{label}_{group_names[COND]}_{NOISE_TYPE}.png"
    # if label == 'group': save_path = os.path.join(error_fitting_group_subfolder, fig_name)
    # else: save_path = os.path.join(error_fitting_sub_subfolder, fig_name)
    
    #plot_FC_matrices(FCemp, FCsim, title1="FCemp", title2="FCsim", size=1, dpi=300)
    
    # fig_name = f"Diff_a{A_FITTING}_N{NPARCELLS}_{label}_{group_names[COND]}_{NOISE_TYPE}.png"
    # if label == 'group': save_path = os.path.join(error_fitting_group_subfolder, fig_name)
    # else: save_path = os.path.join(error_fitting_sub_subfolder, fig_name)
    
    #plot_FC_matrix(FCsim-FCemp, title="diff FCsim-FCemp", size=1.1, dpi=300)

    ## plot the sigma
    # fig_name = f"sigma_fit_a{A_FITTING}_N_{NPARCELLS}_{label}_{group_names[COND]}_{NOISE_TYPE}.png"
    # if label == 'group': save_path = os.path.join(error_fitting_group_subfolder, fig_name)
    # else: save_path = os.path.join(error_fitting_sub_subfolder, fig_name)
    plt.figure(figsize=(np.clip(NPARCELLS, 8, 12), 4))
    plt.plot(range(1, NPARCELLS+1), sigma_ini, '.--', color='gray', alpha=0.5, label='Initial guess')
    plt.plot(range(1, NPARCELLS+1), sigma, '.-', color='tab:blue', alpha=1, label='sigma fit normalized')
    if sigma_2 is not None:
        plt.plot(range(1, NPARCELLS+1), sigma_2, '.-', color='tab:orange', alpha=1, label='sigma fit Adam normalized')
        plt.axhline(np.mean(sigma_2), color='tab:orange', linestyle='--', label=f'{np.mean(sigma_2):.5f}')
    plt.axhline(np.mean(sigma), color='tab:blue', linestyle='--', label=f'{np.mean(sigma_group):.5f}')
    plt.xlabel('Parcels')
    ticks = np.arange(1, NPARCELLS + 1)
    labels = [str(ticks[0])] + [''] * (len(ticks) - 2) + [str(ticks[-1])]
    plt.xticks(ticks, labels)
    plt.legend()
    plt.show()
    # plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    ## plot the a
    a_ini = -0.02 * np.ones(NPARCELLS)
    # fig_name = f"bifur_fit_a{A_FITTING}_N_{NPARCELLS}_{label}_{group_names[COND]}_{NOISE_TYPE}.png"
    # if label == 'group': save_path = os.path.join(error_fitting_group_subfolder, fig_name)
    # else: save_path = os.path.join(error_fitting_sub_subfolder, fig_name)
    plt.figure(figsize=(np.clip(NPARCELLS, 8, 12), 4))
    plt.plot(range(1, NPARCELLS+1), a_ini, '.--', color='gray', alpha=0.5, label='Initial value')
    plt.plot(range(1, NPARCELLS+1), a, '.-', color='tab:blue', alpha=1, label='a fit normalized')
    if a_2 is not None:
        plt.plot(range(1, NPARCELLS+1), a_2, '.-', color='tab:orange', alpha=1, label='a fit Adam normalized')
        plt.axhline(np.mean(a_2), color='tab:orange', linestyle='--', label=f'{np.mean(a_2):.5f}')
    plt.axhline(np.mean(a), color='tab:red', linestyle='--', label=f'{np.mean(a):.5f}')
    plt.xlabel('Parcels')
    ticks = np.arange(1, NPARCELLS + 1)
    labels = [str(ticks[0])] + [''] * (len(ticks) - 2) + [str(ticks[-1])]
    plt.xticks(ticks, labels)
    plt.legend()
    plt.show()
    # plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

NPARCELLS = 40 # max 379
CEFF_FITTING = True
SIGMA_FITTING = True
A_FITTING = True


###### Loading the data ######

DL = ADNI_A.ADNI_A(normalizeBurden=False)

# Loading the timeseries data for all subjects and dividing them into groups
HC_IDs = DL.get_groupSubjects('HC')
HC_MRI = {}
HC_SC = {}
HC_ABeta = []
HC_Tau = []
for subject in HC_IDs:
    data = DL.get_subjectData(subject,printInfo=False)
    HC_MRI[subject] = data[subject]['timeseries'].T
    HC_SC[subject] = data[subject]['SC']
    HC_ABeta.append(np.vstack(data[subject]['ABeta'])) 
    HC_Tau.append(np.vstack(data[subject]['Tau']))

MCI_IDs = DL.get_groupSubjects('MCI')
MCI_MRI = {}
MCI_SC = {}
MCI_ABeta = []
MCI_Tau = []
for subject in MCI_IDs:
    data = DL.get_subjectData(subject,printInfo=False)
    MCI_MRI[subject] = data[subject]['timeseries'].T
    MCI_SC[subject] = data[subject]['SC']
    MCI_ABeta.append(np.vstack(data[subject]['ABeta']))
    MCI_Tau.append(np.vstack(data[subject]['Tau']))

AD_IDs = DL.get_groupSubjects('AD')
AD_MRI = {}
AD_SC = {}
AD_ABeta = []
AD_Tau = []
for subject in AD_IDs:
    data = DL.get_subjectData(subject,printInfo=False)
    AD_MRI[subject] = data[subject]['timeseries'].T
    AD_SC[subject] = data[subject]['SC']
    AD_ABeta.append(np.vstack(data[subject]['ABeta']))
    AD_Tau.append(np.vstack(data[subject]['Tau']))

group_names = ['HC', 'MCI', 'AD']
group_sizes = {'HC': len(HC_IDs), 'MCI': len(MCI_IDs), 'AD': len(AD_IDs)}
a_list_group = []
a_list_sub = []
a_list_group_adam = []
a_list_sub_adam = []

### Prepare the PET data
# Use only the first 360 regions as the subcortical regions do not have PET data
protein_index = min(NPARCELLS,360)
ABeta_burden = [np.array(HC_ABeta)[:,:protein_index,0], np.array(MCI_ABeta)[:,:NPARCELLS,0], np.array(AD_ABeta)[:,:NPARCELLS,0]]
Tau_burden = [np.array(HC_Tau)[:,:protein_index,0], np.array(MCI_Tau)[:,:NPARCELLS,0], np.array(AD_Tau)[:,:NPARCELLS,0]]

### Set parameters
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
sigma_ini = sigma_mean * np.ones(NPARCELLS)
if SIGMA_FITTING: NOISE_TYPE = 'hetero'
else: NOISE_TYPE = 'homo'
COMPETITIVE_COUPLING = False
CEFF_NORMALIZATION = True
maxC = 0.2
iter_check_group = 100
fit_Ceff=CEFF_FITTING
competitive_coupling=COMPETITIVE_COUPLING
fit_sigma=SIGMA_FITTING
sigma_reset=False
Ceff_norm=CEFF_NORMALIZATION
maxC=maxC
iter_check=iter_check_group

## Learning rate settings
epsFC_Ceff = 4e-4
epsCOVtau_Ceff = 1e-4
epsFC_sigma = 8e-5
epsCOVtau_sigma = 3e-5
lrs_Ceff = 1e-3
lrs_sigma = 1e-1
lrs_a = 5e-4
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
MAXiter = 10000
error_tol = 1e-3
patience = 3
learning_rate_factor = 1.0

# Calculate the mean SC matrices per group
HC_SC_matrices = np.array(list(HC_SC.values()))
HC_SC_avg = np.mean(HC_SC_matrices, axis=0)
MCI_SC_matrices = np.array(list(MCI_SC.values()))  # Shape: (Nsubjects, NPARCELLS, NPARCELLS)
MCI_SC_avg = np.mean(MCI_SC_matrices, axis=0)
AD_SC_matrices = np.array(list(AD_SC.values()))  # Shape: (Nsubjects, NPARCELLS, NPARCELLS)
AD_SC_avg = np.mean(AD_SC_matrices, axis=0)

####### Group level #######

# for lr_sig in lrs_sigma:
#     for lr_a in lrs_a:
#         for lr_Ceff in lrs_Ceff:
#             error = 0

TSemp_zsc_list = [] # store the zscored TS for each group
            # Ceff_group_list = [] # store the fitted Ceff for each group
            # sigma_group_list = [] # store the fitted sigma for each group
            # Ceff_group_list_adam = [] # store the fitted Ceff for each group
            # sigma_group_list_adam = [] # store the fitted sigma for each group
for COND in range(3):
    if COND == 0: ## --> HC
        f_diff = calc_H_freq(HC_MRI, 3000, filterps.FiltPowSpetraVersion.v2021)[0]
        ts_gr = HC_MRI
        ID = HC_IDs
        SC = HC_SC_avg  # Use the average SC of the HC group

    elif COND == 1: ## --> MCI
        f_diff = calc_H_freq(MCI_MRI, 3000, filterps.FiltPowSpetraVersion.v2021)[0]
        ts_gr = MCI_MRI
        ID = MCI_IDs
        SC = MCI_SC_avg  # Use the average SC of the MCI group

    elif COND == 2: ## --> AD
        f_diff = calc_H_freq(AD_MRI, 3000, filterps.FiltPowSpetraVersion.v2021)[0]
        ts_gr = AD_MRI
        ID = AD_IDs
        SC = AD_SC_avg  # Use the average SC of the AD group
    
    f_diff = f_diff[:NPARCELLS] # frequencies of group
    omega = 2 * np.pi * f_diff

            #     ### Generates a "group" TS with the same length for all subjects
    min_ntimes = min(ts_gr[subj_id].shape[0] for subj_id in ID)
    ts_gr_arr = np.zeros((len(ID), NPARCELLS, min_ntimes))
    for sub in range(len(ID)):
        subj_id = ID[sub]
        ts_gr_arr[sub,:,:] = ts_gr[subj_id][:min_ntimes,:NPARCELLS].T.copy() 
    TSemp_zsc = zscore_time_series(ts_gr_arr, mode='global', detrend=True)[:,:NPARCELLS,:].copy() #mode: parcel, global, none
    TSemp_zsc_list.append(TSemp_zsc)
            #     SC_N = SC[:NPARCELLS, :NPARCELLS]
            #     SC_N /= np.max(SC_N)
            #     SC_N *= 0.2
            #     Ceff_ini = SC_N.copy()
            #     start_time = time.time()
            #     Ceff_group, sigma_group, a_group, FCemp_group, FCsim_group, error_iter_group, errorFC_iter_group, errorCOVtau_iter_group, = \
            #                                 LinHopf_Ceff_sigma_a_fitting_numba(TSemp_zsc, Ceff_ini, NPARCELLS, TR, f_diff, sigma_ini, Tau=Tau,
            #                                 fit_Ceff=fit_Ceff, competitive_coupling=competitive_coupling, 
            #                                 fit_sigma=SIGMA_FITTING, sigma_reset=sigma_reset,
            #                                 fit_a=A_FITTING,
            #                                 epsFC_Ceff=epsFC_Ceff, epsCOVtau_Ceff=epsCOVtau_Ceff, epsFC_sigma=epsFC_sigma, epsCOVtau_sigma=epsCOVtau_sigma,
            #                                 MAXiter=MAXiter, error_tol=error_tol, patience=patience, learning_rate_factor=learning_rate_factor,
            #                                 Ceff_norm=Ceff_norm, maxC=maxC,
            #                                 iter_check=iter_check, plot_evol=False, plot_evol_last=False)
            #     Ceff_group_adam, sigma_group_adam, a_group_adam, FCemp_group_adam, FCsim_group_adam, error_iter_group_adam, errorFC_iter_group_adam, errorCOVtau_iter_group_adam, = \
            #                                 LinHopf_Ceff_sigma_a_fitting_adam(TSemp_zsc, Ceff_ini, NPARCELLS, TR, f_diff, sigma_ini, Tau=Tau,
            #                                 fit_Ceff=fit_Ceff, competitive_coupling=competitive_coupling, 
            #                                 fit_sigma=SIGMA_FITTING, sigma_reset=sigma_reset,
            #                                 fit_a=A_FITTING,learning_rate_Ceff=lr_Ceff, learning_rate_sigma=lr_sig, learning_rate_a=lr_a,
            #                                 beta1=beta1, beta2=beta2, epsilon=epsilon,
            #                                 MAXiter=MAXiter, error_tol=error_tol, patience=patience)

            #     end_time = time.time()
                
            #     ## save the results
            #     # a_list_group.append(a_group)
            #     # Ceff_group_list.append(Ceff_group)
            #     # sigma_group_list.append(sigma_group)
            #     a_list_group_adam.append(a_group_adam)
            #     Ceff_group_list_adam.append(Ceff_group_adam)
            #     sigma_group_list_adam.append(sigma_group_adam)
            #     error += error_iter_group_adam[-1]
            # #print('Final error:',  error,'lr_sigma:', lr_sig, 'lr_Ceff:', lr_Ceff, 'lr_a:', lr_a)
            #     #print('sigma_group', sigma_group-sigma_group_adam)

            #     show_error(error_iter_group, error_iter_group_adam, errorFC_iter_group, 
            #             errorFC_iter_group_adam, errorCOVtau_iter_group, 
            #             errorCOVtau_iter_group_adam, sigma_group, sigma_group_adam,
            #             sigma_ini, a_group, a_group_adam, FCemp_group, FCsim_group, 
            #             FCsim_group_adam, label="group")

####### Subject level #######
Ceff_means = []
tot_sub_error_adam = 0
tot_sub_error = 0
for COND in range(3):
    a_list_sub_temp = []
    Ceff_sub_temp = []
    if COND == 0: ## --> HC
        ts_gr = HC_MRI
        ID = HC_IDs
        SCs = HC_SC
    elif COND == 1: ## --> MCI
        ts_gr = MCI_MRI
        ID = MCI_IDs
        SCs = MCI_SC
    elif COND == 2: ## --> AD
        ts_gr = AD_MRI
        ID = AD_IDs
        SCs = AD_SC

    
    Ceff_sub = np.zeros((len(ID), NPARCELLS, NPARCELLS))
    sigma_sub = np.zeros((len(ID), NPARCELLS))
    a_sub = np.zeros((len(ID), NPARCELLS))
    FCemp_sub = np.zeros((len(ID), NPARCELLS, NPARCELLS))
    FCsim_sub = np.zeros((len(ID), NPARCELLS, NPARCELLS))
    error_iter_sub = np.ones((len(ID), 200)) * np.nan
    Ceff_sub_adam = np.zeros((len(ID), NPARCELLS, NPARCELLS))
    sigma_sub_adam = np.zeros((len(ID), NPARCELLS))
    a_sub_adam = np.zeros((len(ID), NPARCELLS))
    FCemp_sub_adam = np.zeros((len(ID), NPARCELLS, NPARCELLS))
    FCsim_sub_adam = np.zeros((len(ID), NPARCELLS, NPARCELLS))
    error_iter_sub_adam = np.ones((len(ID), 200)) * np.nan

    f_diff = calc_H_freq(ts_gr, 3000, filterps.FiltPowSpetraVersion.v2021)[1]
    f_diff = f_diff[:,:NPARCELLS] # frequencies of subjects

     
    for sub in range(len(ID)):
        subj_id = ID[sub]
        omega = 2 * np.pi * f_diff[sub,:NPARCELLS] # omega per subject
        SC_N = SCs[subj_id][:NPARCELLS, :NPARCELLS]
        SC_N /= np.max(SC_N)
        SC_N *= 0.2
        #if SIGMA_FITTING: sigma_ini = sigma_group_list[COND].copy()

        Ceff_sub[sub], sigma_sub[sub], a_sub[sub], FCemp_sub[sub], FCsim_sub[sub], error_iter_sub_aux, errorFC_iter_sub_aux, errorCOVtau_iter_sub_aux = \
                                            LinHopf_Ceff_sigma_a_fitting_numba(TSemp_zsc_list[COND][sub], SC_N, NPARCELLS, TR, f_diff[sub], sigma_ini, Tau=Tau,
                                            fit_Ceff=fit_Ceff, competitive_coupling=competitive_coupling, 
                                            fit_sigma=SIGMA_FITTING, sigma_reset=sigma_reset,fit_a=A_FITTING,
                                            epsFC_Ceff=epsFC_Ceff, epsCOVtau_Ceff=epsCOVtau_Ceff, epsFC_sigma=epsFC_sigma*10, epsCOVtau_sigma=epsCOVtau_sigma*10,
                                            MAXiter=MAXiter, error_tol=error_tol, patience=patience+5, learning_rate_factor=learning_rate_factor,
                                            Ceff_norm=Ceff_norm, maxC=maxC,
                                            iter_check=iter_check, plot_evol=False, plot_evol_last=False)
        Ceff_sub_adam[sub], sigma_sub_adam[sub], a_sub_adam[sub], FCemp_sub_adam[sub], FCsim_sub_adam[sub], error_iter_sub_aux_adam, errorFC_iter_sub_aux_adam, errorCOVtau_iter_sub_aux_adam = \
                                            LinHopf_Ceff_sigma_a_fitting_adam(TSemp_zsc_list[COND][sub], SC_N, NPARCELLS, TR, f_diff[sub], sigma_ini, Tau=Tau,
                                            fit_Ceff=fit_Ceff, competitive_coupling=competitive_coupling, 
                                            fit_sigma=SIGMA_FITTING, sigma_reset=sigma_reset,fit_a=A_FITTING,
                                            learning_rate_Ceff=lrs_Ceff, learning_rate_sigma=lrs_sigma, learning_rate_a=lrs_a,
                                            beta1=beta1, beta2=beta2, epsilon=epsilon,
                                            MAXiter=MAXiter, error_tol=error_tol, patience=patience)
        error_iter_sub[sub, :len(error_iter_sub_aux)] = error_iter_sub_aux

        a_list_sub_temp.append(a_sub[sub])
        Ceff_sub_temp.append(Ceff_sub[sub])
        tot_sub_error += error_iter_sub_aux[-1]
        tot_sub_error_adam += error_iter_sub_aux_adam[-1]
    print('Final error:',  tot_sub_error,'adam:', tot_sub_error_adam)
#         # show_error(error_iter_sub_aux, errorFC_iter_sub_aux, errorCOVtau_iter_sub_aux, sigma_sub[sub], sigma_ini, a_sub[sub], FCemp_sub[sub], FCsim_sub[sub], label=f"subj{sub}")
#     a_list_sub.append(np.array(a_list_sub_temp))
#     Ceff_means.append(np.mean(np.array(Ceff_sub_temp), axis=0))


# ### for plotting FC and Ceff matrices (should be incorporated elsewhere)
# # for i in range(3):
# #     Ceff_group_list = np.array(Ceff_group_list)
# #     Ceff_means = np.array(Ceff_means)
# #     Ceff_diff = Ceff_group_list[i] - Ceff_means[i]
# #     plot_FC_matrix(Ceff_diff, title=f"Ceff diff group-{group_names[i]} minus mean subj", size=1.1, dpi=300)
# #     plot_FC_matrix(Ceff_means[i], title=f"Ceff means sub", size=1.1, dpi=300)
# #     plot_FC_matrix(Ceff_group_list[i], title=f"Ceff means group", size=1.1, dpi=300)


# ##### Fitting a values to PET data #####
# a_sub_cortical = [arr[:, :protein_index] for arr in a_list_sub]   # cortical parcels
# a_sub_subcort = [arr[:, protein_index:] for arr in a_list_sub]    # subcortical parcels (19)
# a_group_cortical = [arr[:protein_index] for arr in a_list_group]
# a_group_subcortical = [arr[protein_index:] for arr in a_list_group]

# out = calc_a_values(a_sub_cortical, a_group_cortical, ABeta_burden, Tau_burden)
# predicted_a = out["predicted_a"]
# predicted_a_group = out["predicted_a_group"]
# if protein_index > 360: 
#     a_sub_recombined = [np.hstack((cort, subc)) for cort, subc in zip(predicted_a, a_sub_subcort)]
#     a_group_recombined = [np.hstack((cort, subc)) for cort, subc in zip(predicted_a_group, a_group_subcortical)]
# else:
#     a_sub_recombined = predicted_a
#     a_group_recombined = predicted_a_group


# results = out["results"]
# coef_matrix = out["coef_matrix"]
# print("Coefficient matrix:\n", coef_matrix)
# print("Statistical results of the fit:\n", results)
