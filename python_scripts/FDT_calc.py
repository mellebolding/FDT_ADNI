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
results_dir = os.path.join(repo_root, 'Result_plots')
Ceff_sigma_subfolder = os.path.join(results_dir, 'Ceff_sigma_results')
FDT_values_subfolder = os.path.join(results_dir, 'FDT_values')
FDT_parcel_subfolder = os.path.join(results_dir, 'FDT_parcel')
FDT_subject_subfolder = os.path.join(results_dir, 'FDT_sub')
Inorm1_group_subfolder = os.path.join(results_dir, 'Inorm1_group')
Inorm2_group_subfolder = os.path.join(results_dir, 'Inorm2_group')
Inorm1_sub_subfolder = os.path.join(results_dir, 'Inorm1_sub')
Inorm2_sub_subfolder = os.path.join(results_dir, 'Inorm2_sub')
import numpy as np
from functions_FDT_numba_v9 import *
from numba import njit, prange, objmode
from functions_FC_v3 import *
from functions_LinHopf_Ceff_sigma_fit_v6 import LinHopf_Ceff_sigma_fitting_numba
from scipy.linalg import solve_continuous_lyapunov
import pandas as pd
import matplotlib.pyplot as plt
from functions_violinplots_WN3_v0 import plot_violins_HC_MCI_AD
import p_values as p_values  # Make sure this is working!
import statannotations_permutation

### Loads data from npz file ######################################
def load_appended_records(filepath, filters=None, verbose=False):
    """
    Loads appended records from an .npz file created by `append_record_to_npz`,
    with optional multi-key filtering.

    Parameters
    ----------
    filepath : str
        Path to the .npz file.
    filters : dict or None
        Dictionary of key-value pairs to match (e.g., {'level': 'group', 'condition': 'COND_A'}).
    verbose : bool
        If True, prints debug info.

    Returns
    -------
    list[dict]
        List of matching records.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File '{filepath}' not found.")

    with np.load(filepath, allow_pickle=True) as data:
        if "records" not in data:
            raise KeyError(f"'records' key not found in {filepath}")
        records = list(data["records"])

    if filters:
        records = [
            rec for rec in records
            if all(rec.get(k) == v for k, v in filters.items())
        ]

    if verbose:
        print(f"[load] Loaded {len(records)} matching record(s) from '{filepath}'.")
        if records:
            print(f"[load] Keys in first record: {list(records[0].keys())}")

    return records

def get_field(records, field, filters=None):
    """
    Extract list of values for `field` from records,
    optionally filtering by `filters` dict.
    """
    if filters:
        filtered = [r for r in records if all(r.get(k) == v for k, v in filters.items())]
    else:
        filtered = records
    return [r[field] for r in filtered if field in r]

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

def clear_npz_file(folder, filename):
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)
    # Save an empty records array, overwriting any existing file
    np.savez(filepath, records=np.array([], dtype=object))

###################################################################

def FDT_group_Itmax_norm1_norm2(sigma_group, Ceff_group, omega, a_param, gconst=1.0, v0bias=0.0, tfinal=200, dt=0.01, tmax=100, ts0=0):
    
    Ndim = len(omega[1,:])
    #avec = a_param * np.ones(Ndim)
    I_FDT_all = np.full((3, Ndim), np.nan)
    Inorm1_tmax_s0_group = np.zeros((3, Ndim))
    Inorm2_tmax_s0_group = np.zeros((3, Ndim))

    for COND in range(1, 4):
        avec = a_param[COND-1]
        sigma_group_2 = np.append(sigma_group[COND-1], sigma_group[COND-1])
        v0std = sigma_group_2

        Gamma = -construct_matrix_A(avec, omega[COND-1], Ceff_group[COND-1], gconst)

        v0 = v0std * np.random.standard_normal(2*Ndim) + v0bias
        vsim, noise = Integrate_Langevin_ND_Optimized(Gamma, sigma_group_2, initcond=v0, duration=tfinal, integstep=dt)

        v0 = vsim[:,-1]
        vsim, noise = Integrate_Langevin_ND_Optimized(Gamma, sigma_group_2, initcond=v0, duration=tfinal, integstep=dt)
            
        D = np.diag(sigma_group_2**2 * np.ones(2*Ndim))
        V_0 = solve_continuous_lyapunov(Gamma, D)

        I_tmax_s0 = Its_Langevin_ND(Gamma, sigma_group_2, V_0, tmax, ts0)[0:Ndim]


        I_FDT_all[COND-1, :] = I_tmax_s0
        Inorm1_tmax_s0_group[COND-1] = Its_norm1_Langevin_ND(Gamma, sigma_group_2, V_0, tmax, ts0)[0:Ndim]
        Inorm2_tmax_s0_group[COND-1] = Its_norm2_Langevin_ND(Gamma, sigma_group_2, V_0, tmax, ts0)[0:Ndim]

    return I_FDT_all, Inorm1_tmax_s0_group, Inorm2_tmax_s0_group

def FDT_sub_Itmax_norm1_norm2(sigma_subs, Ceff_subs, omega_subs, a_param, gconst=1.0, v0bias=0.0, tfinal=200, dt=0.01, tmax=100, ts0=0):
    
    Ndim = omega_subs[0].shape[1]
    max_len_subs = max(a.shape[0] for a in omega_subs)
    #print("max_len_subs: ", max_len_subs)
    #print("Ndim: ", Ndim)
    #avec = a_param * np.ones(Ndim)
    I_FDT_all = np.full((3, max_len_subs,Ndim), np.nan)
    Inorm1_tmax_s0_subs = np.full((3, max_len_subs,Ndim), np.nan)
    Inorm2_tmax_s0_subs = np.full((3, max_len_subs,Ndim), np.nan)
    
    index_a = 0
    for COND in range(1, 4):
        for sub in range(sigma_subs[COND-1].shape[0]):
            avec = a_param[index_a]
            index_a += 1
            sigma_subs_2 = np.append(sigma_subs[COND-1][sub, :], sigma_subs[COND-1][sub, :])
            v0std = sigma_subs_2
            
            Gamma = -construct_matrix_A(avec, omega_subs[COND-1][sub, :], Ceff_subs[COND-1][sub, :], gconst)

            v0 = v0std * np.random.standard_normal(2*Ndim) + v0bias
            vsim, noise = Integrate_Langevin_ND_Optimized(Gamma, sigma_subs_2, initcond=v0, duration=tfinal, integstep=dt)

            v0 = vsim[:,-1]
            vsim, noise = Integrate_Langevin_ND_Optimized(Gamma, sigma_subs_2, initcond=v0, duration=tfinal, integstep=dt)
                
            D = np.diag(sigma_subs_2**2 * np.ones(2*Ndim))
            V_0 = solve_continuous_lyapunov(Gamma, D)


            I_tmax_s0 = Its_Langevin_ND(Gamma, sigma_subs_2, V_0, tmax, ts0)[0:Ndim]


            I_FDT_all[COND-1, sub, :] = I_tmax_s0
            Inorm1_tmax_s0_subs[COND-1, sub, :] = Its_norm1_Langevin_ND(Gamma, sigma_subs_2, V_0, tmax, ts0)[0:Ndim]
            Inorm2_tmax_s0_subs[COND-1, sub, :] = Its_norm2_Langevin_ND(Gamma, sigma_subs_2, V_0, tmax, ts0)[0:Ndim]
    return I_FDT_all, Inorm1_tmax_s0_subs, Inorm2_tmax_s0_subs

def X_group_Itmax_norm1_norm2(sigma_group, Ceff_group, omega_group, NPARCELLS, a_param=-0.02, gconst=1.0):
    tmax = 5000
    ts = 0
    intR_tmax_s0_group = np.zeros((3, NPARCELLS))
    intRnorm1_tmax_s0_group = np.zeros((3, NPARCELLS))
    intRnorm2_tmax_s0_group = np.zeros((3, NPARCELLS))
    for COND in range(3):
        avec = a_param[COND]
        sigma_vec = np.append(sigma_group[COND], sigma_group[COND])
        Gamma = -construct_matrix_A(avec, omega_group[COND], Ceff_group[COND], gconst)
        D = np.diag(sigma_vec**2 * np.ones(2*NPARCELLS))
        V_0 = solve_continuous_lyapunov(Gamma, D)
        intR_tmax_s0_group[COND] = intRts_Langevin_ND(Gamma, tmax, ts)[0:NPARCELLS]
        intRnorm1_tmax_s0_group[COND] = intRts_norm1_Langevin_ND(Gamma, 10000, ts)[0:NPARCELLS]
        intRnorm2_tmax_s0_group[COND] = intRts_norm2_Langevin_ND(Gamma, sigma_vec, V_0, tmax, ts)[0:NPARCELLS]

    return intR_tmax_s0_group, intRnorm1_tmax_s0_group, intRnorm2_tmax_s0_group

def X_sub_Itmax_norm1_norm2(sigma_subs, Ceff_subs, omega_subs, NPARCELLS, a_param=-0.02, gconst=1.0):
    tmax = 5000
    ts = 0
    a_index = 0
    max_len_subs = max(a.shape[0] for a in omega_subs)
    intR_tmax_s0_subject = np.full((3, max_len_subs,NPARCELLS), np.nan)
    intRnorm1_tmax_s0_subject = np.full((3, max_len_subs,NPARCELLS), np.nan)
    intRnorm2_tmax_s0_subject = np.full((3, max_len_subs,NPARCELLS), np.nan)

    for COND in range(3):
        for sub in range(sigma_subs[COND].shape[0]):
            avec = a_param[a_index]
            a_index += 1
            sigma_vec = np.append(sigma_subs[COND][sub, :], sigma_subs[COND][sub, :])
            Gamma = -construct_matrix_A(avec, omega_subs[COND][sub, :], Ceff_subs[COND][sub, :], gconst)
            D = np.diag(sigma_vec**2 * np.ones(2*NPARCELLS))
            V_0 = solve_continuous_lyapunov(Gamma, D)
            intR_tmax_s0_subject[COND, sub, :] = intRts_Langevin_ND(Gamma, tmax, ts)[0:NPARCELLS]
            intRnorm1_tmax_s0_subject[COND, sub, :] = intRts_norm1_Langevin_ND(Gamma, 10000, ts)[0:NPARCELLS]
            intRnorm2_tmax_s0_subject[COND, sub, :] = intRts_norm2_Langevin_ND(Gamma, sigma_vec, V_0, tmax, ts)[0:NPARCELLS]

    return intR_tmax_s0_subject, intRnorm1_tmax_s0_subject, intRnorm2_tmax_s0_subject
####################################################################

#### Base parameters ####
NPARCELLS = 379
NOISE_TYPE = "Hetero"
A_FITTING = False

### Load the Ceff and sigma fitting results ###
if A_FITTING:
    all_records = load_appended_records(
    filepath=os.path.join(Ceff_sigma_subfolder, f"Ceff_sigma_a{A_FITTING}_N{NPARCELLS}_{NOISE_TYPE}.npz")
    )
    savefilename = f"FDT_values_a{A_FITTING}_N{NPARCELLS}_{NOISE_TYPE}.npz"
else:
    all_records = load_appended_records(
    filepath=os.path.join(Ceff_sigma_subfolder, f"Ceff_sigma_{NPARCELLS}_{NOISE_TYPE}.npz")
    #filepath=os.path.join(Ceff_sigma_subfolder, f"Ceff_sigma_aFalse_N{NPARCELLS}_{NOISE_TYPE}.npz")
    )
    savefilename = f"FDT_values_N{NPARCELLS}_{NOISE_TYPE}.npz"

clear_npz_file(FDT_values_subfolder, savefilename)


# Load all records
print('done loading')
# Extract group-level data
HC_group_sig = np.array(get_field(all_records, "sigma", filters={"level": "group", "condition": "0"}))
HC_group_Ceff = np.array(get_field(all_records, "Ceff", filters={"level": "group", "condition": "0"}))
HC_group_omega = np.array(get_field(all_records, "omega", filters={"level": "group", "condition": "0"}))
MCI_group_sig = np.array(get_field(all_records, "sigma", filters={"level": "group", "condition": "1"}))
MCI_group_Ceff = np.array(get_field(all_records, "Ceff", filters={"level": "group", "condition": "1"}))
MCI_group_omega = np.array(get_field(all_records, "omega", filters={"level": "group", "condition": "1"}))
AD_group_sig = np.array(get_field(all_records, "sigma", filters={"level": "group", "condition": "2"}))
AD_group_Ceff = np.array(get_field(all_records, "Ceff", filters={"level": "group", "condition": "2"}))
AD_group_omega = np.array(get_field(all_records, "omega", filters={"level": "group", "condition": "2"}))
print(MCI_group_sig)
sigma_group = np.array([HC_group_sig[0], MCI_group_sig[0], AD_group_sig[0]])
Ceff_group = np.array([HC_group_Ceff[0], MCI_group_Ceff[0], AD_group_Ceff[0]])
omega = np.array([HC_group_omega[0], MCI_group_omega[0], AD_group_omega[0]])

# Extract subject-level data
HC_subs_sig = np.array(get_field(all_records, "sigma", filters={"level": "subject", "condition": "0"}))
HC_subs_Ceff = np.array(get_field(all_records, "Ceff", filters={"level": "subject", "condition": "0"}))
HC_subs_omega = np.array(get_field(all_records, "omega", filters={"level": "subject", "condition": "0"}))
MCI_subs_sig = np.array(get_field(all_records, "sigma", filters={"level": "subject", "condition": "1"}))
MCI_subs_Ceff = np.array(get_field(all_records, "Ceff", filters={"level": "subject", "condition": "1"}))
MCI_subs_omega = np.array(get_field(all_records, "omega", filters={"level": "subject", "condition": "1"}))
AD_subs_sig = np.array(get_field(all_records, "sigma", filters={"level": "subject", "condition": "2"}))
AD_subs_Ceff = np.array(get_field(all_records, "Ceff", filters={"level": "subject", "condition": "2"}))
AD_subs_omega = np.array(get_field(all_records, "omega", filters={"level": "subject", "condition": "2"}))

# lists, as the arrays are not the same length
sigma_subs = [HC_subs_sig, MCI_subs_sig, AD_subs_sig]
Ceff_subs = [HC_subs_Ceff, MCI_subs_Ceff, AD_subs_Ceff]
omega_subs = [HC_subs_omega, MCI_subs_omega, AD_subs_omega]

# Extract a parameters if available
if A_FITTING:
    a_group = np.vstack(get_field(all_records, "a", filters={"level": "group"}))
    a_subs = np.vstack(get_field(all_records, "a", filters={"level": "subject"}))
    a_group_org = np.array(get_field(all_records, "original_a", filters={"level": "group"}))
    a_subs_org = get_field(all_records, "original_a", filters={"level": "subject"})
else:
    a_group = np.array([-0.02, -0.02, -0.02])
    a_subs = np.array([-0.02] * HC_subs_sig.shape[0] + [-0.02] * MCI_subs_sig.shape[0] + [-0.02] * AD_subs_sig.shape[0])
    a_group_org = -0.02
    a_subs_org = -0.02

### Calculate FDT values (could probably do without tmax & N1 here)###
# group analysis 
I_tmax_group,I_norm1_group,I_norm2_group = FDT_group_Itmax_norm1_norm2(sigma_group, Ceff_group, omega, a_group, gconst=1.0, v0bias=0.0, tfinal=200, dt=0.01, tmax=100, ts0=0)
X_I_tmax_group, X_Inorm1_group, X_Inorm2_group = X_group_Itmax_norm1_norm2(sigma_group, Ceff_group, omega, NPARCELLS, a_group, gconst=1.0)

# subject analysis
I_tmax_sub, I_norm1_sub, I_norm2_sub = FDT_sub_Itmax_norm1_norm2(sigma_subs, Ceff_subs, omega_subs, a_subs, gconst=1.0, v0bias=0.0, tfinal=200, dt=0.01, tmax=100, ts0=0)
X_I_tmax_sub, X_I_norm1_sub, X_I_norm2_sub = X_sub_Itmax_norm1_norm2(sigma_subs, Ceff_subs, omega_subs, NPARCELLS, a_subs, gconst=1.0)

append_record_to_npz(
    FDT_values_subfolder,
    savefilename,
    level="subject",
    I_tmax = I_tmax_sub,
    I_norm1 = I_norm1_sub,
    I_norm2 = I_norm2_sub,
    X_I_tmax = X_I_tmax_sub,
    X_Inorm1 = X_I_norm1_sub,
    X_Inorm2 = X_I_norm2_sub,
    a = a_subs,
    original_a = a_subs_org,
)
append_record_to_npz(
    FDT_values_subfolder,
    savefilename,
    level="group",
    I_tmax = I_tmax_group,
    I_norm1 = I_norm1_group,
    I_norm2 = I_norm2_group,
    X_I_tmax = X_I_tmax_group,
    X_Inorm1 = X_Inorm1_group,
    X_Inorm2 = X_Inorm2_group,
    a = a_group,
    original_a = a_group_org,
)