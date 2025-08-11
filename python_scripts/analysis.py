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
import numpy as np
from functions_FDT_numba_v9 import *
from numba import njit, prange, objmode
from functions_FC_v3 import *
from functions_LinHopf_Ceff_sigma_fit_v6 import LinHopf_Ceff_sigma_fitting_numba
from scipy.linalg import solve_continuous_lyapunov

### Loads data from npz file ######################################

import os
import numpy as np

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
    
    Ndim = len(omega)
    avec = a_param * np.ones(Ndim)
    print(avec)
    I_FDT_all = np.full((3, Ndim), np.nan)
    Inorm1_tmax_s0_group = np.zeros((3, Ndim))
    Inorm2_tmax_s0_group = np.zeros((3, Ndim))
    #print(avec.shape)
    for COND in range(1, 4):

        sigma_group_2 = np.append(sigma_group[COND-1], sigma_group[COND-1])
        v0std = sigma_group_2
        
        Gamma = -construct_matrix_A(avec, omega[COND-1], Ceff_group[COND-1], gconst)

        v0 = v0std * np.random.standard_normal(Ndim) + v0bias
        vsim, noise = Integrate_Langevin_ND_Optimized(Gamma, sigma_group_2, initcond=v0, duration=tfinal, integstep=dt)

        v0 = vsim[:,-1]
        vsim, noise = Integrate_Langevin_ND_Optimized(Gamma, sigma_group_2, initcond=v0, duration=tfinal, integstep=dt)
            
        D = np.diag(sigma_group_2**2 * np.ones(Ndim))
        V_0 = solve_continuous_lyapunov(Gamma, D)


        I_tmax_s0 = Its_Langevin_ND(Gamma, sigma_group_2, V_0, tmax, ts0)[0:Ndim]


        I_FDT_all[COND-1, :] = I_tmax_s0
        Inorm1_tmax_s0_group[COND-1] = Its_norm1_Langevin_ND(Gamma, sigma_group_2, V_0, tmax, ts0)[0:Ndim]
        Inorm2_tmax_s0_group[COND-1] = Its_norm2_Langevin_ND(Gamma, sigma_group_2, V_0, tmax, ts0)[0:Ndim]

    return I_FDT_all, Inorm1_tmax_s0_group, Inorm2_tmax_s0_group

NPARCELLS = 18
NOISE_TYPE = "HOMO"

# Load all records
all_records = load_appended_records(
    filepath=os.path.join(Ceff_sigma_subfolder, f"Ceff_sigma_{NPARCELLS}_{NOISE_TYPE}.npz")
)

HC_group_sig = np.array(get_field(all_records, "sigma", filters={"level": "group", "condition": "1"}))
HC_group_Ceff = np.array(get_field(all_records, "Ceff", filters={"level": "group", "condition": "1"}))
HC_group_omega = np.array(get_field(all_records, "omega", filters={"level": "group", "condition": "1"}))
MCI_group_sig = np.array(get_field(all_records, "sigma", filters={"level": "group", "condition": "2"}))
MCI_group_Ceff = np.array(get_field(all_records, "Ceff", filters={"level": "group", "condition": "2"}))
MCI_group_omega = np.array(get_field(all_records, "omega", filters={"level": "group", "condition": "2"}))
AD_group_sig = np.array(get_field(all_records, "sigma", filters={"level": "group", "condition": "3"}))
AD_group_Ceff = np.array(get_field(all_records, "Ceff", filters={"level": "group", "condition": "3"}))
AD_group_omega = np.array(get_field(all_records, "omega", filters={"level": "group", "condition": "3"}))

print(AD_group_omega.shape)
print(AD_group_Ceff.shape)
#print(avec.shape)
sigma_group = np.array([HC_group_sig[0], MCI_group_sig[0], AD_group_sig[0]])
Ceff_group = np.array([HC_group_Ceff[0], MCI_group_Ceff[0], AD_group_Ceff[0]])
omega = np.array([HC_group_omega[0], MCI_group_omega[0], AD_group_omega[0]])

x,y,z = FDT_group_Itmax_norm1_norm2(sigma_group, Ceff_group, omega, a_param=-0.02, gconst=1.0, v0bias=0.0, tfinal=200, dt=0.01, tmax=100, ts0=0)

print("x: ",x, "\ny: ",y, "\nz: ",z)
