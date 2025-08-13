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

def figures_I_tmax_norm1_norm2(group, subject, I_tmax, I_norm1, I_norm2):
    group_names = ['HC', 'MCI', 'AD']
    records_parcel_Itmax = []
    records_parcel_norm1 = []
    records_parcel_norm2 = []
    records_subject_Itmax = []
    records_subject_norm1 = []
    records_subject_norm2 = []

    if group:
        I_tmax_group = I_tmax
        I_norm1_group = I_norm1
        I_norm2_group = I_norm2

        for group_idx, group_name in enumerate(group_names):
            for parcel in range(I_tmax_group.shape[0]):
                records_parcel_Itmax.append({
                "value": I_tmax_group[group_idx, parcel],
                "cond": group_name,
                "parcel": parcel
                })
                records_parcel_norm1.append({
                "value": I_norm1_group[group_idx, parcel],
                "cond": group_name,
                "parcel": parcel
                })
                records_parcel_norm2.append({
                "value": I_norm2_group[group_idx, parcel],
                "cond": group_name,
                "parcel": parcel
                })
        data_parcels_Itmax = pd.DataFrame.from_records(records_parcel_Itmax)
        data_parcels_norm1 = pd.DataFrame.from_records(records_parcel_norm1)
        data_parcels_norm2 = pd.DataFrame.from_records(records_parcel_norm2)
        resI_Itmax = {
            'HC': data_parcels_Itmax[data_parcels_Itmax['cond'] == 'HC']['value'].values,
            'MCI': data_parcels_Itmax[data_parcels_Itmax['cond'] == 'MCI']['value'].values,
            'AD': data_parcels_Itmax[data_parcels_Itmax['cond'] == 'AD']['value'].values,
        }
        resI_norm1 = {
            'HC': data_parcels_norm1[data_parcels_norm1['cond'] == 'HC']['value'].values,
            'MCI': data_parcels_norm1[data_parcels_norm1['cond'] == 'MCI']['value'].values,
            'AD': data_parcels_norm1[data_parcels_norm1['cond'] == 'AD']['value'].values,
        }
        resI_norm2 = {
            'HC': data_parcels_norm2[data_parcels_norm2['cond'] == 'HC']['value'].values,
            'MCI': data_parcels_norm2[data_parcels_norm2['cond'] == 'MCI']['value'].values,
            'AD': data_parcels_norm2[data_parcels_norm2['cond'] == 'AD']['value'].values,
        }

        plt.rcParams.update({'font.size': 15})
        fig_name = f"box_parcel_Itmax_N{NPARCELLS}_{NOISE_TYPE}"
        save_path = os.path.join(FDT_parcel_subfolder, fig_name)
        p_values.plotComparisonAcrossLabels2(
            resI_Itmax,
            custom_test=statannotations_permutation.stat_permutation_test,
            columnLables=['HC', 'MCI', 'AD'],
            graphLabel='FDT I(tmax, 0) Parcels',
            save_path=save_path
        )
        plt.rcParams.update({'font.size': 15})
        fig_name = f"box_parcel_norm1_N{NPARCELLS}_{NOISE_TYPE}"
        save_path = os.path.join(Inorm1_group_subfolder, fig_name)
        p_values.plotComparisonAcrossLabels2(
            resI_norm1,
            custom_test=statannotations_permutation.stat_permutation_test,
            columnLables=['HC', 'MCI', 'AD'],
            graphLabel='FDT I Norm1 Parcels',
            save_path=save_path
        )
        plt.rcParams.update({'font.size': 15})
        fig_name = f"box_parcel_norm2_N{NPARCELLS}_{NOISE_TYPE}"
        save_path = os.path.join(Inorm2_group_subfolder, fig_name)
        p_values.plotComparisonAcrossLabels2(
            resI_norm2,
            custom_test=statannotations_permutation.stat_permutation_test,
            columnLables=['HC', 'MCI', 'AD'],
            graphLabel='FDT I Norm2 Parcels',
            save_path=save_path
        )

    if subject:
        I_tmax_sub = I_tmax
        I_norm1_sub = I_norm1
        I_norm2_sub = I_norm2
        I_tmax_sub_mean = np.nanmean(I_tmax_sub, axis=2)
        I_norm1_sub_mean = np.nanmean(I_norm1_sub, axis=2)
        I_norm2_sub_mean = np.nanmean(I_norm2_sub, axis=2)

        for groupidx, group_name in enumerate(group_names):
            for subject in range(I_tmax_sub_mean.shape[1]):
                records_subject_Itmax.append({
                    "value": I_tmax_sub_mean[groupidx, subject],
                    "cond": group_name,
                    "subject": subject
                })
                records_subject_norm1.append({
                    "value": I_norm1_sub_mean[groupidx, subject],
                    "cond": group_name,
                    "subject": subject
                })
                records_subject_norm2.append({
                    "value": I_norm2_sub_mean[groupidx, subject],
                    "cond": group_name,
                    "subject": subject
                })
        
        data_subjects_Itmax = pd.DataFrame.from_records(records_subject_Itmax)
        data_subjects_norm1 = pd.DataFrame.from_records(records_subject_norm1)
        data_subjects_norm2 = pd.DataFrame.from_records(records_subject_norm2)

        fig, ax = plt.subplots(figsize=(10, 10))
        fig_name = f"violin_subject_N{NPARCELLS}_{NOISE_TYPE}"
        save_path = os.path.join(FDT_subject_subfolder, fig_name)
        plot_violins_HC_MCI_AD(
            ax=ax,
            data=data_subjects_Itmax,
            font_scale=1.4,
            metric='I(t=tmax,s=0) [Subject mean]',
            point_size=5,
            xgrid=False,
            plot_title='FDT I(tmax, 0) — Mean per subject per group',
            saveplot=1,
            filename=save_path,
            dpi=300
        )
        fig, ax = plt.subplots(figsize=(10, 10))
        fig_name = f"violin_subject_norm1_N{NPARCELLS}_{NOISE_TYPE}"
        save_path = os.path.join(Inorm1_sub_subfolder, fig_name)
        plot_violins_HC_MCI_AD(
            ax=ax,
            data=data_subjects_norm1,
            font_scale=1.4,
            metric='I Norm1 [Subject mean]',
            point_size=5,
            xgrid=False,
            plot_title='FDT I Norm1 — Mean per subject per group',
            saveplot=1,
            filename=save_path,
            dpi=300
        )
        fig, ax = plt.subplots(figsize=(10, 10))
        fig_name = f"violin_subject_norm2_N{NPARCELLS}_{NOISE_TYPE}"
        save_path = os.path.join(Inorm2_sub_subfolder, fig_name)
        plot_violins_HC_MCI_AD(
            ax=ax,
            data=data_subjects_norm2,
            font_scale=1.4,
            metric='I Norm2 [Subject mean]',
            point_size=5,
            xgrid=False,
            plot_title='FDT I Norm2 — Mean per subject per group',
            saveplot=1,
            filename=save_path,
            dpi=300
        )

def figures_barplot_parcels(option,I_tmax_group):
    if option == 'I_tmax':
        I_group = I_tmax_group
    elif option == 'I_norm1':
        I_group = I_norm1_group
    elif option == 'I_norm2':
        I_group = I_norm2_group
    else:
        raise ValueError("Invalid option. Choose from 'I_tmax', 'I_norm1', or 'I_norm2'.")

    colors = ['tab:blue', 'tab:orange', 'tab:green']

    plt.figure(figsize=(12, 6))
    fig_name = f"barplot_parcel_{option}_N{NPARCELLS}_{NOISE_TYPE}"
    save_path = os.path.join(FDT_parcel_subfolder, fig_name)
    bottom = np.zeros(18)  # start at zero for stacking
    for i in range(3):
        plt.bar(range(18), I_group[i], bottom=bottom, color=colors[i], label=f'{["HC", "MCI", "AD"][i]}', alpha=0.7)
        bottom += I_group[i]

    plt.xlabel('Parcel')
    plt.ylabel(f'{option}')
    plt.title(f'{option} for Parcels')
    plt.legend()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_means_per_RSN(name, I_tmax_group, NPARCELLS=18):

    group_names = ['HC', 'MCI', 'AD']

    # Compute mean per RSN for each group
    means_per_group = []
    for g in range(I_tmax_group.shape[0]):
        group_means = []
        for rsn_name, nodes in RSNs.items():
            nodes_in_range = [n for n in nodes if n < NPARCELLS]
            if nodes_in_range:  # avoid empty
                group_means.append(np.nanmean(I_tmax_group[g, nodes_in_range]))
            else:
                group_means.append(np.nan)
        means_per_group.append(group_means)

    means_per_group = np.array(means_per_group)
    fig_name = f"barplot_RSN_{name}_N{NPARCELLS}_{NOISE_TYPE}"
    save_path = os.path.join(FDT_parcel_subfolder, fig_name)
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(RSNs))
    width = 0.25

    for i, group in enumerate(group_names):
        ax.bar(x + i*width - width, means_per_group[i], width, label=group)

    ax.set_xticks(x)
    ax.set_xticklabels(RSNs.keys(), rotation=45)
    ax.set_ylabel(f'Mean {name}')
    ax.set_title(f'Mean {name} per RSN (first {NPARCELLS} parcels)')
    ax.legend()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_means_per_subjects_per_RSN(RSN, I_tmax_sub, nameRSN, nameI, NPARCELLS):
    subjects_per_group = [17, 9, 10]   # number of valid subjects per group

    nodes_in_range = [n for n in RSN if n < NPARCELLS]

    group_names = ['HC', 'MCI', 'AD']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # distinct for each group

    means = []
    labels = []
    bar_colors = []
    group_avg_positions = []
    group_avg_values = []

    pos = 0
    for g, group in enumerate(group_names):
        n_subs = subjects_per_group[g]
        group_vals = []
        for s in range(n_subs):
            mean_val = np.nanmean(I_tmax_sub[g, s, nodes_in_range])
            means.append(mean_val)
            group_vals.append(mean_val)
            labels.append(f'{group}_S{s+1}')
            bar_colors.append(colors[g])
        group_avg_values.append(np.nanmean(group_vals))
        group_avg_positions.append((pos, pos + n_subs - 1))
        pos += n_subs

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    fig_name = f"barplot_{nameRSN}_sub_{nameI}_N{NPARCELLS}_{NOISE_TYPE}"
    save_path = os.path.join(FDT_subject_subfolder, fig_name)
    ax.bar(range(len(means)), means, color=bar_colors)
    ax.set_xticks(range(len(means)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_ylabel(f'Mean {nameI}')
    ax.set_title(f'Mean {nameI} for {nameRSN} RSN')

    # Group separators
    for g, (start, end) in enumerate(group_avg_positions):
        width = (end - start) + 0.8   # span exactly over all subject bars
        ax.bar(
            start - 0.8/2,            # left edge aligns with first subject bar
            group_avg_values[g],            # height
            width=width,                     # covers the group’s bars
            color=colors[g],
            alpha=0.6,                       # transparency
            edgecolor='black',
            linewidth=1,
            align='edge'                     # align by left edge, not center
        )
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
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

#figures_I_tmax_norm1_norm2(group=True, subject=False, I_tmax=I_tmax_group, I_norm1=I_norm1_group, I_norm2=I_norm2_group)
#figures_I_tmax_norm1_norm2(group=False, subject=True, I_tmax=I_tmax_sub, I_norm1=I_norm1_sub, I_norm2=I_norm2_sub)

#figures_barplot_parcels('I_tmax',I_tmax_group)
#figures_barplot_parcels('I_norm1', I_norm1_group)
#figures_barplot_parcels('I_norm2', I_norm2_group)


##### RESTING STATE NETWORKS #####
SomMot = [7, 8, 23, 35, 38, 39, 40, 42, 50, 52, 54, 55, 56, 98, 99, 100, 101, 102, 103, 104, 105, 106, 114, 123, 124, 167, 172, 173, 174, 187, 188, 191, 203, 207, 215, 218, 219, 220, 221, 230, 232, 233, 234, 235, 279, 280, 281, 282, 283, 284, 303, 347, 352, 353, 354]
Vis = [0, 1, 2, 3, 4, 5, 6, 12, 15, 17, 18, 19, 20, 21, 22, 118, 119, 120, 125, 126, 141, 142, 145, 151, 152, 153, 154, 155, 156, 157, 158, 159, 162, 180, 181, 182, 183, 184, 185, 186, 192, 195, 198, 199, 200, 201, 202, 300, 322, 331, 332, 333, 335, 337, 338, 339, 342]
Def = [25, 27, 29, 30, 32, 33, 34, 60, 63, 64, 65, 67, 68, 70, 71, 73, 74, 75, 86, 87, 93, 122, 127, 128, 129, 130, 131, 148, 149, 150, 160, 175, 176, 177, 205, 209, 210, 212, 213, 214, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 254, 255, 256, 266, 268, 273, 277, 286, 291, 298, 299, 302, 304, 305, 306, 307, 308, 309, 311, 313, 318, 321, 328, 329, 330, 334, 340, 341, 344, 349, 355, 356, 359]
DorsAttn = [9, 16, 26, 28, 41, 44, 45, 46, 47, 48, 49, 51, 53, 79, 95, 115, 116, 135, 136, 137, 139, 140, 144, 189, 190, 196, 197, 224, 225, 226, 227, 228, 229, 231, 259, 275, 276, 295, 296, 315, 316, 317, 319, 320, 324, 325, 336]
Cont = [13, 14, 57, 61, 62, 66, 69, 72, 76, 78, 80, 81, 82, 83, 84, 90, 94, 96, 97, 110, 132, 143, 161, 169, 170, 178, 179, 193, 194, 208, 237, 252, 253, 258, 260, 261, 262, 263, 264, 265, 270, 274, 290, 312, 323, 350]
Limbic = [88, 89, 91, 92, 117, 121, 133, 134, 163, 164, 165, 171, 267, 269, 271, 272, 297, 301, 310, 314, 343, 345, 351]
SalVentAttn = [10, 11, 24, 31, 36, 37, 43, 58, 59, 77, 85, 107, 108, 109, 111, 112, 113, 138, 146, 147, 166, 168, 204, 206, 211, 216, 217, 222, 223, 236, 238, 239, 257, 278, 285, 287, 288, 289, 292, 293, 294, 326, 327, 346, 348, 357, 358]

RSNs = {
    'SomMot': SomMot,
    'Vis': Vis,
    'Def': Def,
    'DorsAttn': DorsAttn,
    'Cont': Cont,
    'Limbic': Limbic,
    'SalVentAttn': SalVentAttn
}

#plot_means_per_RSN('I_tmax', I_tmax_group, NPARCELLS=NPARCELLS)
#plot_means_per_RSN('I_norm1', I_norm1_group, NPARCELLS=NPARCELLS)
#plot_means_per_RSN('I_norm2', I_norm2_group, NPARCELLS=NPARCELLS)

#plot_means_per_subjects_per_RSN(SomMot, I_tmax_sub, 'SomMot', 'I_tmax', NPARCELLS)
#plot_means_per_subjects_per_RSN(Vis, I_tmax_sub, 'Vis', 'I_tmax', NPARCELLS)
# ...


###### VISUALIZATION ######
from nilearn import surface, datasets, plotting
import nibabel as nib
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# -----------------------
# Load your parcellation
# -----------------------
# For example, your parcellation NIfTI (replace with your file)
nii_path = os.path.join('ADNI-A_DATA', 'MNI_Glasser_HCP_v1.0.nii.gz')
parcel_img = nib.load(nii_path)
parcel_data = parcel_img.get_fdata().astype(int)
affine = parcel_img.affine

# Initialize new labeled array
labeled_img = np.zeros(parcel_data.shape, dtype=int)

# Right hemisphere: labels 1-180
for label in range(1, 181):
    labeled_img[parcel_data == label] = label

# Left hemisphere: labels 1001-1180 -> map to 181-360
for idx, label in enumerate(range(1001, 1181), start=181):
    labeled_img[parcel_data == label] = idx

labeled_img = labeled_img.astype(np.int32)
# Save new NIfTI
out_path = os.path.join('ADNI-A_DATA', 'MNI_Glasser_HCP_v1.0_labeled_360.nii.gz')
new_img = nib.Nifti1Image(labeled_img, affine)
nib.save(new_img, out_path)

print("Saved labeled 360-parcel image to:", out_path)
print("Unique labels:", np.unique(labeled_img))
# Alternative: read directly from dataobj
# parcel_data = np.array(parcel_img.dataobj, dtype=np.int32)

# Check unique labels
# labels = np.unique(parcel_data)
# print(f"Unique labels in the parcellation: {labels}")
# print(f"Shape: {parcel_data.shape}, Non-zero voxels: {np.count_nonzero(parcel_data)}")
# #parcel_data = np.asanyarray(parcel_img.dataobj)
# #parcel_data = parcel_img.get_fdata()


# print("Shape:", parcel_data.shape)
# print("Data type:", parcel_data.dtype)
# print("Min/Max values:", parcel_data.min(), parcel_data.max())
# print("Affine:\n", parcel_img.affine)


#print(parcel_data)
# Your group-level values (one per parcel)
# e.g., 360 parcels for Glasser360
parcel_img = nib.load(out_path)
parcel_data = parcel_img.get_fdata().astype(int)

group_values = I_tmax_group
group_maps = []

# Create a volume where each voxel gets the parcel's value
for g in range(3):
    group_map = np.zeros_like(parcel_data)
    
    # Only fill the first 18 parcels, leave the rest as 0
    for i, val in enumerate(group_values[g]):
        group_map[parcel_data == (i + 1)] = val
    
    group_maps.append(group_map)

group_img = nib.Nifti1Image(group_map.astype(np.float32), affine=parcel_img.affine)
#group_img = nib.Nifti1Image(group_map, affine=parcel_img.affine)

# -----------------------
# Fetch fsaverage surface
# -----------------------
fsaverage = datasets.fetch_surf_fsaverage()

# Project volume to surface
texture_left = surface.vol_to_surf(group_img, fsaverage.pial_left)
texture_right = surface.vol_to_surf(group_img, fsaverage.pial_right)

# Global color limits
vmin = np.min(group_values)
vmax = np.max(group_values)

# -----------------------
# Create surface plots
# -----------------------
fig = plt.figure(figsize=(10, 5))

# Left hemisphere
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
plotting.plot_surf_stat_map(
    fsaverage.pial_left, texture_left,
    hemi='left', view='lateral',
    colorbar=False, cmap='viridis',
    bg_map=fsaverage.sulc_left,
    vmin=vmin, vmax=vmax,
    axes=ax1, title='Left', darkness=None
)

# Right hemisphere
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
plotting.plot_surf_stat_map(
    fsaverage.pial_right, texture_right,
    hemi='right', view='lateral',
    colorbar=False, cmap='viridis',
    bg_map=fsaverage.sulc_right,
    vmin=vmin, vmax=vmax,
    axes=ax2, title='Right', darkness=None
)

# -----------------------
# Add colorbar
# -----------------------
norm = Normalize(vmin=vmin, vmax=vmax)
sm = ScalarMappable(cmap='viridis', norm=norm)
sm.set_array([])

cbar_ax = fig.add_axes([0.47, 0.25, 0.02, 0.5])  # adjust position
cbar = plt.colorbar(sm, cax=cbar_ax)
cbar.set_label("Group Difference")

plt.tight_layout()
plt.show()

print("Unique labels in final image:", np.unique(labeled_img))
print("Non-zero voxel count:", np.count_nonzero(labeled_img))
