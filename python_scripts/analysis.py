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
    return [r.get(field) for r in filtered]


###################################################################

def FDT_group_Itmax_norm1_norm2(sigma_group, Ceff_group, omega, a_param=-0.02, gconst=1.0, v0bias=0.0, tfinal=200, dt=0.01, tmax=100, ts0=0):
    
    Ndim = len(omega[1,:])
    avec = a_param * np.ones(Ndim)
    I_FDT_all = np.full((3, Ndim), np.nan)
    Inorm1_tmax_s0_group = np.zeros((3, Ndim))
    Inorm2_tmax_s0_group = np.zeros((3, Ndim))

    for COND in range(1, 4):

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

def FDT_sub_Itmax_norm1_norm2(sigma_subs, Ceff_subs, omega_subs, a_param=-0.02, gconst=1.0, v0bias=0.0, tfinal=200, dt=0.01, tmax=100, ts0=0):
    
    Ndim = omega_subs[0].shape[1]
    max_len_subs = max(a.shape[0] for a in omega_subs)
    #print("max_len_subs: ", max_len_subs)
    #print("Ndim: ", Ndim)
    avec = a_param * np.ones(Ndim)
    I_FDT_all = np.full((3, max_len_subs,Ndim), np.nan)
    Inorm1_tmax_s0_subs = np.full((3, max_len_subs,Ndim), np.nan)
    Inorm2_tmax_s0_subs = np.full((3, max_len_subs,Ndim), np.nan)
    
    for COND in range(1, 4):
        for sub in range(sigma_subs[COND-1].shape[0]):

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
####################################################################

NPARCELLS = 18
NOISE_TYPE = "HOMO"


# Load all records
all_records = load_appended_records(
    filepath=os.path.join(Ceff_sigma_subfolder, f"Ceff_sigma_{NPARCELLS}_{NOISE_TYPE}.npz")
)

# Extract group-level data
HC_group_sig = np.array(get_field(all_records, "sigma", filters={"level": "group", "condition": "1"}))
HC_group_Ceff = np.array(get_field(all_records, "Ceff", filters={"level": "group", "condition": "1"}))
HC_group_omega = np.array(get_field(all_records, "omega", filters={"level": "group", "condition": "1"}))
MCI_group_sig = np.array(get_field(all_records, "sigma", filters={"level": "group", "condition": "2"}))
MCI_group_Ceff = np.array(get_field(all_records, "Ceff", filters={"level": "group", "condition": "2"}))
MCI_group_omega = np.array(get_field(all_records, "omega", filters={"level": "group", "condition": "2"}))
AD_group_sig = np.array(get_field(all_records, "sigma", filters={"level": "group", "condition": "3"}))
AD_group_Ceff = np.array(get_field(all_records, "Ceff", filters={"level": "group", "condition": "3"}))
AD_group_omega = np.array(get_field(all_records, "omega", filters={"level": "group", "condition": "3"}))

sigma_group = np.array([HC_group_sig[0], MCI_group_sig[0], AD_group_sig[0]])
Ceff_group = np.array([HC_group_Ceff[0], MCI_group_Ceff[0], AD_group_Ceff[0]])
omega = np.array([HC_group_omega[0], MCI_group_omega[0], AD_group_omega[0]])

# Extract subject-level data
HC_subs_sig = np.array(get_field(all_records, "sigma", filters={"level": "subject", "condition": "1"}))
HC_subs_Ceff = np.array(get_field(all_records, "Ceff", filters={"level": "subject", "condition": "1"}))
HC_subs_omega = np.array(get_field(all_records, "omega", filters={"level": "subject", "condition": "1"}))
MCI_subs_sig = np.array(get_field(all_records, "sigma", filters={"level": "subject", "condition": "2"}))
MCI_subs_Ceff = np.array(get_field(all_records, "Ceff", filters={"level": "subject", "condition": "2"}))
MCI_subs_omega = np.array(get_field(all_records, "omega", filters={"level": "subject", "condition": "2"}))
AD_subs_sig = np.array(get_field(all_records, "sigma", filters={"level": "subject", "condition": "3"}))
AD_subs_Ceff = np.array(get_field(all_records, "Ceff", filters={"level": "subject", "condition": "3"}))
AD_subs_omega = np.array(get_field(all_records, "omega", filters={"level": "subject", "condition": "3"}))

# lists, as the arrays are not the same length
sigma_subs = [HC_subs_sig, MCI_subs_sig, AD_subs_sig]
Ceff_subs = [HC_subs_Ceff, MCI_subs_Ceff, AD_subs_Ceff]
omega_subs = [HC_subs_omega, MCI_subs_omega, AD_subs_omega]

# group analysis
I_tmax_group,I_norm1_group,I_norm2_group = FDT_group_Itmax_norm1_norm2(sigma_group, Ceff_group, omega, a_param=-0.02, gconst=1.0, v0bias=0.0, tfinal=200, dt=0.01, tmax=100, ts0=0)

# subject analysis
I_tmax_sub, I_norm1_sub, I_norm2_sub = FDT_sub_Itmax_norm1_norm2(sigma_subs, Ceff_subs, omega_subs, a_param=-0.02, gconst=1.0, v0bias=0.0, tfinal=200, dt=0.01, tmax=100, ts0=0)

print("I_tmax_group.shape", I_tmax_group.shape, 
      "I_norm1_group.shape", I_norm1_group.shape, 
      "I_norm2_group.shape", I_norm2_group.shape,
      "I_tmax_sub.shape", I_tmax_sub.shape,
      "I_norm1_sub.shape", I_norm1_sub.shape,
      "I_norm2_sub.shape", I_norm2_sub.shape)


group_names = ['HC', 'MCI', 'AD']
records_parcel = []
records_subject = []

print("I_tmax_sub", I_tmax_sub, "I_norm1_sub", I_norm1_sub, "I_norm2_sub", I_norm2_sub)

I_tmax_group_mean = np.nanmean(I_tmax_group, axis=1) # this are 3 numbers..
I_norm1_group_mean = np.nanmean(I_norm1_group, axis=1)
I_norm2_group_mean = np.nanmean(I_norm2_group, axis=1)
I_tmax_sub_mean = np.nanmean(I_tmax_sub, axis=2)
I_norm1_sub_mean = np.nanmean(I_norm1_sub, axis=2)
I_norm2_sub_mean = np.nanmean(I_norm2_sub, axis=2)

print("I_tmax_sub_mean", I_tmax_sub_mean)
print("I_tmax_group_mean", I_tmax_group)
for group_idx, group_name in enumerate(group_names):
    for parcel in range(I_tmax_group.shape[0]):
        records_parcel.append({
            "value": I_tmax_group[group_idx, parcel],
            "cond": group_name,
            "parcel": parcel
        })

for groupidx, group_name in enumerate(group_names):
    for subject in range(I_tmax_sub_mean.shape[1]):
        records_subject.append({
            "value": I_tmax_sub_mean[groupidx, subject],
            "cond": group_name,
            "subject": subject
        })

data_parcels = pd.DataFrame.from_records(records_parcel)
data_subjects = pd.DataFrame.from_records(records_subject)

fig, ax = plt.subplots(figsize=(10, 10))
fig_name = f"violin_subject_N{NPARCELLS}_{NOISE_TYPE}"
save_path = os.path.join(FDT_subject_subfolder, fig_name)
plot_violins_HC_MCI_AD(
    ax=ax,
    data=data_subjects,
    font_scale=1.4,
    metric='I(t=tmax,s=0) [Subject mean]',
    point_size=5,
    xgrid=False,
    plot_title='FDT I(tmax, 0) — Mean per subject per group',
    saveplot=1,
    filename=save_path,
    dpi=300
)
print(records_subject)
print(data_subjects)
