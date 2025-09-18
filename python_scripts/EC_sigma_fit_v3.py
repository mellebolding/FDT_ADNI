import os
import sys


#### Setting up paths ####

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
#ECgroup_subfolder = os.path.join(results_dir, 'EC_group')
Ceff_sigma_subfolder = os.path.join(results_dir, 'Ceff_sigma_results')
#ECsub_subfolder = os.path.join(results_dir, 'EC_sub')
#FCgroup_subfolder = os.path.join(results_dir, 'FC_group')
#FCsub_subfolder = os.path.join(results_dir, 'FC_sub')
#sigma_subfolder = os.path.join(results_dir, 'sig_sub')
#sigma_group_subfolder = os.path.join(results_dir, 'sig_group')
#FDT_parcel_subfolder = os.path.join(results_dir, 'FDT_parcel')
#FDT_subject_subfolder = os.path.join(results_dir, 'FDT_sub')
#Inorm1_group_subfolder = os.path.join(results_dir, 'Inorm1_group')
#Inorm2_group_subfolder = os.path.join(results_dir, 'Inorm2_group')
#Inorm1_sub_subfolder = os.path.join(results_dir, 'Inorm1_sub')
#Inorm2_sub_subfolder = os.path.join(results_dir, 'Inorm2_sub')
#training_dir = os.path.join(results_dir, 'training_conv')
error_fitting_group_subfolder = os.path.join(results_dir, 'error_fitting_group')
error_fitting_sub_subfolder = os.path.join(results_dir, 'error_fitting_sub')
os.makedirs(results_dir, exist_ok=True)
# os.makedirs(ECgroup_subfolder, exist_ok=True)
# os.makedirs(ECsub_subfolder, exist_ok=True)
# os.makedirs(FCgroup_subfolder, exist_ok=True)
os.makedirs(Ceff_sigma_subfolder, exist_ok=True)
# os.makedirs(FCsub_subfolder, exist_ok=True)
# os.makedirs(sigma_subfolder, exist_ok=True)
# os.makedirs(sigma_group_subfolder, exist_ok=True)
# os.makedirs(FDT_parcel_subfolder, exist_ok=True)
# os.makedirs(FDT_subject_subfolder, exist_ok=True)
# os.makedirs(Inorm1_group_subfolder, exist_ok=True)
# os.makedirs(Inorm2_group_subfolder, exist_ok=True)
# os.makedirs(Inorm1_sub_subfolder, exist_ok=True)
# os.makedirs(Inorm2_sub_subfolder, exist_ok=True)
# os.makedirs(training_dir, exist_ok=True)
os.makedirs(error_fitting_group_subfolder, exist_ok=True)
os.makedirs(error_fitting_sub_subfolder, exist_ok=True)


#### Importing necessary packages and functions ####

from functions_FDT_numba_v9 import *
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.signal import detrend as scipy_detrend
from functions_FC_v3 import *
from functions_LinHopf_Ceff_sigma_fit_v6 import LinHopf_Ceff_sigma_fitting_numba
from function_LinHopf_Ceff_sigma_a_fit import LinHopf_Ceff_sigma_a_fitting_numba
from function_LinHopf_Ceff_sigma_a_fit import from_PET_to_a_global
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
from scipy.linalg import expm
import pandas as pd
import scipy.integrate as integrate
from scipy.linalg import solve_continuous_lyapunov
from DataLoaders.baseDataLoader import DataLoader
import ADNI_A
from functions_boxplots_WN3_v0 import *
from functions_violinplots_WN3_v0 import *
from functions_violinplots_v2 import *
from functions_FDT_numba_v9 import construct_matrix_A, Integrate_Langevin_ND_Optimized, closest_valid_M
import filterps
import functions_boxplots_WN3_v0
from typing import Union
from numba import njit, prange, objmode
import time
import p_values as p_values
import statannotations_permutation
from LinHopf_EC_Sig_A_fit_adam_numba import LinHopf_Ceff_sigma_a_fitting_adam


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
    """
    Clears the contents of a .npz file by overwriting it with an empty 'records' array.
    """
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

def show_error(error_iter, errorFC_iter, errorCOVtau_iter, sigma, sigma_ini, a, FCemp, FCsim, label):
    """
    Want to give an indication of the fitting quality?
    options to show: final error, FC fit, COVtau fit, sigma fit, a fit
    """
    
    if error_iter is not None:
        figure_name = f"error_iter_a{A_FITTING}_N{NPARCELLS}_{label}_{group_names[COND]}_{NOISE_TYPE}.png"
        if label == 'group': save_path = os.path.join(error_fitting_group_subfolder, figure_name)
        else: save_path = os.path.join(error_fitting_sub_subfolder, figure_name)
        plt.figure(figsize=(8,5))
        plt.plot(np.arange(1, len(error_iter) + 1) * 100, error_iter, 'o-', color='tab:blue', label='Error @100 iter')
        plt.plot(np.arange(1, len(errorFC_iter) + 1) * 100, errorFC_iter, 's-', color='tab:orange', label='Error FC @100 iter')
        plt.plot(np.arange(1, len(errorCOVtau_iter) + 1) * 100, errorCOVtau_iter, '^-', color='tab:green', label='Error COVtau @100 iter')
        plt.xlabel('Iteration')
        plt.ylabel('Error')
        plt.title(f"Error Curves - Group {group_names[COND]}")
        plt.legend(loc='upper right', fontsize=10)
        plt.grid(True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        

    ## plotting the FC and Ceff matrices
    fig_name = f"FCmatrices_a{A_FITTING}_N{NPARCELLS}_{label}_{group_names[COND]}_{NOISE_TYPE}.png"
    if label == 'group': save_path = os.path.join(error_fitting_group_subfolder, fig_name)
    else: save_path = os.path.join(error_fitting_sub_subfolder, fig_name)
    plot_FC_matrices(FCemp, FCsim, title1="FCemp", title2="FCsim", save_path=save_path, size=1, dpi=300)
    fig_name = f"Diff_a{A_FITTING}_N{NPARCELLS}_{label}_{group_names[COND]}_{NOISE_TYPE}.png"
    if label == 'group': save_path = os.path.join(error_fitting_group_subfolder, fig_name)
    else: save_path = os.path.join(error_fitting_sub_subfolder, fig_name)
    plot_FC_matrix(FCsim-FCemp, title="diff FCsim-FCemp", size=1.1, save_path=save_path,dpi=300)

    ## plot the sigma
    fig_name = f"sigma_fit_a{A_FITTING}_N_{NPARCELLS}_{label}_{group_names[COND]}_{NOISE_TYPE}.png"
    if label == 'group': save_path = os.path.join(error_fitting_group_subfolder, fig_name)
    else: save_path = os.path.join(error_fitting_sub_subfolder, fig_name)
    plt.figure(figsize=(np.clip(NPARCELLS, 8, 12), 4))
    plt.plot(range(1, NPARCELLS+1), sigma_ini, '.--', color='gray', alpha=0.5, label='Initial guess')
    plt.plot(range(1, NPARCELLS+1), sigma, '.-', color='tab:blue', alpha=1, label='sigma fit normalized')
    plt.axhline(np.mean(sigma), color='tab:blue', linestyle='--', label=f'{np.mean(sigma_group):.5f}')
    plt.xlabel('Parcels')
    ticks = np.arange(1, NPARCELLS + 1)
    labels = [str(ticks[0])] + [''] * (len(ticks) - 2) + [str(ticks[-1])]
    plt.xticks(ticks, labels)
    plt.legend()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    ## plot the a
    a_ini = -0.02 * np.ones(NPARCELLS)
    fig_name = f"bifur_fit_a{A_FITTING}_N_{NPARCELLS}_{label}_{group_names[COND]}_{NOISE_TYPE}.png"
    if label == 'group': save_path = os.path.join(error_fitting_group_subfolder, fig_name)
    else: save_path = os.path.join(error_fitting_sub_subfolder, fig_name)
    plt.figure(figsize=(np.clip(NPARCELLS, 8, 12), 4))
    plt.plot(range(1, NPARCELLS+1), a_ini, '.--', color='gray', alpha=0.5, label='Initial value')
    plt.plot(range(1, NPARCELLS+1), a, '.-', color='tab:blue', alpha=1, label='a fit normalized')
    plt.axhline(np.mean(a), color='tab:red', linestyle='--', label=f'{np.mean(a):.5f}')
    plt.xlabel('Parcels')
    ticks = np.arange(1, NPARCELLS + 1)
    labels = [str(ticks[0])] + [''] * (len(ticks) - 2) + [str(ticks[-1])]
    plt.xticks(ticks, labels)
    plt.legend()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
#############################################################


##### Base parameters settings ######

NPARCELLS = 45 # max 379
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

### Prepare the PET data
# Use only the first 360 regions as the subcortical regions do not have PET data
protein_index = min(NPARCELLS,360)
ABeta_burden = [np.array(HC_ABeta)[:,:protein_index,0], np.array(MCI_ABeta)[:,:NPARCELLS,0], np.array(AD_ABeta)[:,:NPARCELLS,0]]
Tau_burden = [np.array(HC_Tau)[:,:protein_index,0], np.array(MCI_Tau)[:,:NPARCELLS,0], np.array(AD_Tau)[:,:NPARCELLS,0]]


### Set parameters
Tau = 2
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
COMPETITIVE_COUPLING = True
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
lr_Ceff = 1e-3
lr_sigma = 1e-2
lr_a = 5e-4
beta1 = 0.85
beta2 = 0.995
epsilon = 1e-8
MAXiter = 10000
error_tol = 1e-3
patience = 6
learning_rate_factor = 1.0

# Clear the previous file
clear_npz_file(Ceff_sigma_subfolder, f"Ceff_sigma_a{A_FITTING}_N{NPARCELLS}_{NOISE_TYPE}.npz")

# Calculate the mean SC matrices per group
HC_SC_matrices = np.array(list(HC_SC.values()))
HC_SC_avg = np.mean(HC_SC_matrices, axis=0)
MCI_SC_matrices = np.array(list(MCI_SC.values()))  # Shape: (Nsubjects, NPARCELLS, NPARCELLS)
MCI_SC_avg = np.mean(MCI_SC_matrices, axis=0)
AD_SC_matrices = np.array(list(AD_SC.values()))  # Shape: (Nsubjects, NPARCELLS, NPARCELLS)
AD_SC_avg = np.mean(AD_SC_matrices, axis=0)


####### Group level #######
TSemp_zsc_list = [] # store the zscored TS for each group
Ceff_group_list = [] # store the fitted Ceff for each group
sigma_group_list = [] # store the fitted sigma for each group
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

    ### Generates a "group" TS with the same length for all subjects
    min_ntimes = min(ts_gr[subj_id].shape[0] for subj_id in ID)
    ts_gr_arr = np.zeros((len(ID), NPARCELLS, min_ntimes))
    for sub in range(len(ID)):
        subj_id = ID[sub]
        ts_gr_arr[sub,:,:] = ts_gr[subj_id][:min_ntimes,:NPARCELLS].T.copy() 
    TSemp_zsc = zscore_time_series(ts_gr_arr, mode='global', detrend=True)[:,:NPARCELLS,:].copy() #mode: parcel, global, none
    TSemp_zsc_list.append(TSemp_zsc)
    SC_N = SC[:NPARCELLS, :NPARCELLS]
    SC_N /= np.max(SC_N)
    SC_N *= 0.2
    Ceff_ini = SC_N.copy()

    start_time = time.time()
 
    Ceff_group, sigma_group, a_group, FCemp_group, FCsim_group, error_iter_group, errorFC_iter_group, errorCOVtau_iter_group, = \
                                LinHopf_Ceff_sigma_a_fitting_adam(TSemp_zsc, Ceff_ini, NPARCELLS, TR, f_diff, sigma_ini, Tau=Tau,
                                            fit_Ceff=fit_Ceff, competitive_coupling=competitive_coupling, 
                                            fit_sigma=SIGMA_FITTING, sigma_reset=sigma_reset,
                                            fit_a=A_FITTING,learning_rate_Ceff=lr_Ceff, learning_rate_sigma=lr_sigma, learning_rate_a=lr_a,
                                            beta1=beta1, beta2=beta2, epsilon=epsilon,
                                            MAXiter=MAXiter, error_tol=error_tol, patience=patience)

    end_time = time.time()
    
    ## save the results
    a_list_group.append(a_group)
    Ceff_group_list.append(Ceff_group)
    sigma_group_list.append(sigma_group)
    print('Final error:',  error_iter_group[-1], 'Condition:', COND, 'Time (s):', end_time - start_time)
    #print('sigma_group', sigma_group)

    append_record_to_npz(
    Ceff_sigma_subfolder,
    f"Ceff_sigma_a{A_FITTING}_N{NPARCELLS}_{NOISE_TYPE}.npz",
    level="group",
    condition=f"{COND}",
    sigma=sigma_group,
    Ceff=Ceff_group,
    omega=omega)

    show_error(error_iter_group, errorFC_iter_group, errorCOVtau_iter_group, sigma_group, sigma_ini, a_group, FCemp_group, FCsim_group, label="group")


####### Subject level #######
Ceff_means = []
for COND in range(3):
    a_list_sub_temp = []
    Ceff_sub_temp = []
    tot_sub_error = 0
    if COND == 0: ## --> HC
        ts_gr = HC_MRI
        ID = HC_IDs
        SCs = HC_SC
        EC_gr = Ceff_group_list[COND]
    elif COND == 1: ## --> MCI
        ts_gr = MCI_MRI
        ID = MCI_IDs
        SCs = MCI_SC
        EC_gr = Ceff_group_list[COND]
    elif COND == 2: ## --> AD
        ts_gr = AD_MRI
        ID = AD_IDs
        SCs = AD_SC
        EC_gr = Ceff_group_list[COND]

    
    Ceff_sub = np.zeros((len(ID), NPARCELLS, NPARCELLS))
    sigma_sub = np.zeros((len(ID), NPARCELLS))
    a_sub = np.zeros((len(ID), NPARCELLS))
    FCemp_sub = np.zeros((len(ID), NPARCELLS, NPARCELLS))
    FCsim_sub = np.zeros((len(ID), NPARCELLS, NPARCELLS))
    error_iter_sub = np.ones((len(ID), 200)) * np.nan

    f_diff = calc_H_freq(ts_gr, 3000, filterps.FiltPowSpetraVersion.v2021)[1]
    f_diff = f_diff[:,:NPARCELLS] # frequencies of subjects


    for sub in range(len(ID)):
        subj_id = ID[sub]
        omega = 2 * np.pi * f_diff[sub,:NPARCELLS] # omega per subject
        # SC_N = SCs[subj_id][:NPARCELLS, :NPARCELLS]
        # SC_N /= np.max(SC_N)
        # SC_N *= 0.2
        SC_N = EC_gr
        if SIGMA_FITTING: sigma_ini = sigma_group_list[COND].copy()

        Ceff_sub[sub], sigma_sub[sub], a_sub[sub], FCemp_sub[sub], FCsim_sub[sub], error_iter_sub_aux, errorFC_iter_sub_aux, errorCOVtau_iter_sub_aux = \
                                            LinHopf_Ceff_sigma_a_fitting_adam(TSemp_zsc_list[COND][sub], SC_N, NPARCELLS, TR, f_diff[sub], sigma_ini, Tau=Tau,
                                            fit_Ceff=fit_Ceff, competitive_coupling=competitive_coupling, 
                                            fit_sigma=SIGMA_FITTING, sigma_reset=sigma_reset,fit_a=A_FITTING,
                                            learning_rate_Ceff=lr_Ceff, learning_rate_sigma=lr_sigma, learning_rate_a=lr_a,
                                            beta1=beta1, beta2=beta2, epsilon=epsilon,
                                            MAXiter=MAXiter, error_tol=error_tol, patience=patience)
        error_iter_sub[sub, :len(error_iter_sub_aux)] = error_iter_sub_aux

        a_list_sub_temp.append(a_sub[sub])
        Ceff_sub_temp.append(Ceff_sub[sub])
        tot_sub_error += error_iter_sub_aux[-1]
        print(error_iter_sub_aux[-1], subj_id, COND)

        append_record_to_npz(
        Ceff_sigma_subfolder,
        f"Ceff_sigma_a{A_FITTING}_N{NPARCELLS}_{NOISE_TYPE}.npz",
        level="subject",
        condition=f"{COND}",
        subject=f"S{sub}",
        sigma=sigma_sub[sub],
        Ceff=Ceff_sub[sub],
        omega=omega)
        show_error(error_iter_sub_aux, errorFC_iter_sub_aux, errorCOVtau_iter_sub_aux, sigma_sub[sub], sigma_ini, a_sub[sub], FCemp_sub[sub], FCsim_sub[sub], label=f"subj{sub}")
    a_list_sub.append(np.array(a_list_sub_temp))
    Ceff_means.append(np.mean(np.array(Ceff_sub_temp), axis=0))
    print('Final error:',  tot_sub_error, 'Condition:', COND)

### for plotting FC and Ceff matrices (should be incorporated elsewhere)
# for i in range(3):
#     Ceff_group_list = np.array(Ceff_group_list)
#     Ceff_means = np.array(Ceff_means)
#     Ceff_diff = Ceff_group_list[i] - Ceff_means[i]
#     plot_FC_matrix(Ceff_diff, title=f"Ceff diff group-{group_names[i]} minus mean subj", size=1.1, dpi=300)
#     plot_FC_matrix(Ceff_means[i], title=f"Ceff means sub", size=1.1, dpi=300)
#     plot_FC_matrix(Ceff_group_list[i], title=f"Ceff means group", size=1.1, dpi=300)


##### Fitting a values to PET data #####
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

append_record_to_npz(
        Ceff_sigma_subfolder,
        f"Ceff_sigma_a{A_FITTING}_N{NPARCELLS}_{NOISE_TYPE}.npz",
        level="subject"#,
        # a = a_sub_recombined,
        # original_a = a_list_sub
        )

append_record_to_npz(
        Ceff_sigma_subfolder,
        f"Ceff_sigma_a{A_FITTING}_N{NPARCELLS}_{NOISE_TYPE}.npz",
        level="group"#,
        # a = a_group_recombined,
        # original_a = np.array(a_list_group)
        )