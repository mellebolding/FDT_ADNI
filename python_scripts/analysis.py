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

def load_appended_records(filepath, filter_key=None, filter_value=None, verbose=False):
    """
    Loads all appended records from an .npz file created by `append_record_to_npz`,
    with optional filtering.

    Parameters
    ----------
    filepath : str
        Path to the .npz file.
    filter_key : str or None
        Key to filter records on (e.g., 'level', 'condition').
    filter_value : any or None
        Value the key must match to include the record.
    verbose : bool
        If True, print debug info.

    Returns
    -------
    list[dict]
        List of matching records (each a dict).
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File '{filepath}' not found.")

    with np.load(filepath, allow_pickle=True) as data:
        if "records" not in data:
            raise KeyError(f"'records' key not found in {filepath}")
        records = list(data["records"])  # Convert numpy array to list of dicts

    if filter_key is not None and filter_value is not None:
        records = [rec for rec in records if rec.get(filter_key) == filter_value]

    if verbose:
        print(f"[load] Loaded {len(records)} record(s) from '{filepath}'.")
        if records:
            print(f"[load] Example keys: {list(records[0].keys())}")

    return records

###################################################################

def FDT_group_Itmax_norm1_norm2(sigma_group, Ceff_group, omega, a_param=-0.02, gconst=1.0, v0bias=0.0, tfinal=200, dt=0.01, tmax=100, ts0=0):
    """
    Calculate FDT group Itmax norm1 and norm2.
    
    Parameters:
    - sigma_group: Standard deviation for the group.
    - Ceff_group: Effective connectivity for the group.
    - omega: Frequency array.
    - avec: Coefficient vector.
    - gconst: Constant for the integration.
    - v0bias: Initial bias for velocity.
    - tfinal: Final time for integration.
    - dt: Time step for integration.
    - group_names: Names of the groups.
    - cond_index_map: Mapping of condition names to indices.
    - I_FDT_all: Array to store FDT results.
    - Inorm1_tmax_s0_group: Array to store norm1 results.
    - Inorm2_tmax_s0_group: Array to store norm2 results.
    - COND: Current condition index.

    Returns:
    None
    """
    
    Ndim = len(omega)
    avec = a_param * np.ones(Ndim)
    group_names = ['HC', 'MCI', 'AD']
    
    # Duplicate sigma_group for two groups
    sigma_group_2 = np.append(sigma_group, sigma_group)
    v0std = sigma_group_2

    
    Gamma = -construct_matrix_A(avec, omega, Ceff_group, gconst)

    v0 = v0std * np.random.standard_normal(Ndim) + v0bias
    vsim, noise = Integrate_Langevin_ND_Optimized(Gamma, sigma_group_2, initcond=v0, duration=tfinal, integstep=dt)

    v0 = vsim[:,-1]
    vsim, noise = Integrate_Langevin_ND_Optimized(Gamma, sigma_group_2, initcond=v0, duration=tfinal, integstep=dt)
        
    D = np.diag(sigma_group_2**2 * np.ones(Ndim))
    V_0 = solve_continuous_lyapunov(Gamma, D)


    I_tmax_s0 = Its_Langevin_ND(Gamma, sigma_group_2, V_0, tmax, ts0)[0:Ndim]

    group_name = group_names[COND - 1]
    group_idx = cond_index_map[group_name]
    #subject_idx = sub  # Already incrementing

    I_FDT_all[group_idx, :] = I_tmax_s0
    Inorm1_tmax_s0_group[group_idx] = Its_norm1_Langevin_ND(Gamma, sigma_group_2, V_0, tmax, ts0)[0:Ndim]
    Inorm2_tmax_s0_group[group_idx] = Its_norm2_Langevin_ND(Gamma, sigma_group_2, V_0, tmax, ts0)[0:Ndim]

    return I_FDT_all, Inorm1_tmax_s0_group, Inorm2_tmax_s0_group

NPARCELLS = 18
NOISE_TYPE = "HOMO"

filepath = os.path.join(Ceff_sigma_subfolder, f"Ceff_sigma_{NPARCELLS}_{NOISE_TYPE}.npz")

# Load all records
all_records = load_appended_records(filepath, verbose=True)

# Load only group-level records
group_records = load_appended_records(filepath, filter_key="level", filter_value="group")
for rec in group_records:
    print(rec["sigma"], rec["Ceff"], rec["omega"])

# Load only subject "sub-01"
subject_records = load_appended_records(filepath, filter_key="subject", filter_value="sub-01")
