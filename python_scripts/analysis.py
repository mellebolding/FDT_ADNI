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
from functions_boxplots_WN3_v0 import plot_boxes_HC_MCI_AD
import p_values as p_values
import statannotations_permutation
from nilearn import surface, datasets, plotting
import nibabel as nib
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import ADNI_A

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



###################################################################
def RSN_significance_group(I_norm2_group,a=False):
    records_norm2 = []
    group_names = ['HC', 'MCI', 'AD']

    for group_idx, group_name in enumerate(group_names):
        for rsn_name, nodes in RSNs.items():
            nodes_in_range = [n for n in nodes if n < I_norm2_group.shape[1]]
            if not nodes_in_range:
                continue
            for parcel in nodes_in_range:
                records_norm2.append({
                    "value": I_norm2_group[group_idx, parcel],
                    "cond": group_name,
                    "parcel": parcel,
                    "rsn": rsn_name
                })

    data_parcels_norm2 = pd.DataFrame.from_records(records_norm2)
    resI_norm2 = {
        rsn_name: {
            'HC': data_parcels_norm2[(data_parcels_norm2['cond'] == 'HC') & (data_parcels_norm2['rsn'] == rsn_name)]['value'].values,
            'MCI': data_parcels_norm2[(data_parcels_norm2['cond'] == 'MCI') & (data_parcels_norm2['rsn'] == rsn_name)]['value'].values,
            'AD': data_parcels_norm2[(data_parcels_norm2['cond'] == 'AD') & (data_parcels_norm2['rsn'] == rsn_name)]['value'].values,
        }
        for rsn_name in RSNs.keys()
    }
    plt.rcParams.update({'font.size': 15})

    for rsn_name, res in resI_norm2.items():
        fig_name = f"box_rsn_{rsn_name}_a{a}_N{NPARCELLS}_{NOISE_TYPE}"
        save_path = os.path.join(FDT_parcel_subfolder, fig_name)

        p_values.plotComparisonAcrossLabels2(
            res,
            custom_test=statannotations_permutation.stat_permutation_test,
            columnLables=['HC', 'MCI', 'AD'],
            graphLabel=f'FDT I_norm2 {rsn_name} {NOISE_TYPE} a{a}',
            save_path=save_path
        )

def RSN_radar_plot(I_norm2_group, a=False):
    group_names = ['HC', 'MCI', 'AD']
    num_rsn = len(RSNs)
    angles = np.linspace(0, 2 * np.pi, num_rsn, endpoint=False).tolist()
    angles += angles[:1]  # complete the loop

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    # --- Plot each group ---
    for group_idx, group_name in enumerate(group_names):
        means = []
        for rsn_name, nodes in RSNs.items():
            nodes_in_range = [n for n in nodes if n < I_norm2_group.shape[1]]
            if nodes_in_range:
                means.append(np.nanmean(I_norm2_group[group_idx, nodes_in_range]))
            else:
                means.append(np.nan)
        means += means[:1]  # complete the loop

        ax.plot(angles, means, label=group_name, linewidth=2)
        ax.fill(angles, means, alpha=0.25)

    # --- RSN names outside ---
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(RSNs.keys(), fontsize=12, weight="bold")
    for label in ax.get_xticklabels():
        label.set_y(label.get_position()[1] - 0.15)  # push outward

    # --- Improve grid lines ---
    ax.set_ylim(0, np.nanmax(means) * 1.3)  # dynamic max limit
    ax.set_yticklabels([])  # hide labels
    ax.grid(True, color="black", alpha=0.3, linewidth=1.2)  # darker & thicker circles

    ax.set_title(f'FDT I_norm2 per RSN {NOISE_TYPE} a{a}', size=15, y=1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

    fig_name = f"radar_rsn_a{a}_N{NPARCELLS}_{NOISE_TYPE}"
    save_path = os.path.join(FDT_parcel_subfolder, fig_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def I_vs_Xnorm2(I_norm2_group, X_norm2_group, I_a=None,X_a=None,a=False, sub=False):
    """
    Scatter plot: I_norm2_group (x-axis) vs X_norm2_group (y-axis) for each group,
    with a linear fit per group + slope and R² in legend.
    """
    group_names = ['HC', 'MCI', 'AD']
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    plt.figure(figsize=(8, 6))
    if a: 
        group_names = ['HC_a', 'MCI_a', 'AD_a']
        colors = ['navy', 'red', 'darkgreen']
    if I_a is not None and X_a is not None:
        I_norm2_group = np.concatenate((I_norm2_group, I_a), axis=0)
        X_norm2_group = np.concatenate((X_norm2_group, X_a), axis=0)
        group_names = ['HC', 'MCI', 'AD', 'HC_a', 'MCI_a', 'AD_a']
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'navy', 'red', 'darkgreen']
    if sub: 
        I_norm2_group = np.nanmean(I_norm2_group, axis=2)
        X_norm2_group = np.nanmean(X_norm2_group, axis=2)
    

    for i, group in enumerate(group_names):
        x = I_norm2_group[i]
        y = X_norm2_group[i]
        mask = ~np.isnan(x) & ~np.isnan(y)
        x = x[mask]
        y = y[mask]

        # scatter
        plt.scatter(x, y, color=colors[i], alpha=0.7)

        # linear fit
        if len(x) > 1:  # need at least 2 points
            coef = np.polyfit(x, y, 1)   # slope + intercept
            poly1d_fn = np.poly1d(coef)

            # line over x-range
            x_line = np.linspace(np.min(x), np.max(x), 100)
            y_line = poly1d_fn(x_line)

            plt.plot(x_line, y_line, color=colors[i], linewidth=2)

            # compute R²
            y_pred = poly1d_fn(x)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - (ss_res / ss_tot if ss_tot > 0 else np.nan)

            # add slope and R² to legend label
            label = f"{group} (slope={coef[0]:.3f}, R²={r2:.2f})"
        else:
            label = f"{group} (insufficient data)"

        plt.scatter([], [], color=colors[i], label=label)  # dummy for legend

    plt.xlabel('I_norm2')
    plt.ylabel('X_norm2')
    if sub: 
        plt.title(f'I_norm2 vs X_norm2 per subject {NOISE_TYPE} a{a}')
    else: plt.title(f'I_norm2 vs X_norm2 per group {NOISE_TYPE} a{a}')
    plt.legend()
    plt.tight_layout()
    plt.show()

from scipy.stats import sem, t

def I_vs_Xnorm22(I_norm2_group, X_norm2_group, a=False):
    """
    Scatter plot: I_norm2_group (x-axis) vs X_norm2_group (y-axis) for each group,
    with subject-level linear fits aggregated into group mean fit + 95% CI.
    Input arrays expected shape: (n_groups, n_subjects, n_parcels).
    """
    group_names = ['HC', 'MCI', 'AD']
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    plt.figure(figsize=(8, 6))
    #i = 2
    for i, group in enumerate(group_names):
        x_group = I_norm2_group[i]   # shape (n_subjects, n_parcels)
        y_group = X_norm2_group[i]

        # scatter all valid subject points
        plt.scatter(x_group.flatten(), y_group.flatten(), color=colors[i], alpha=0.3)

        # x range for fits
        x_min, x_max = np.nanmin(x_group), np.nanmax(x_group)
        x_line = np.linspace(x_min, x_max, 100)

        subj_preds = []
        slopes = []

        for subj_x, subj_y in zip(x_group, y_group):
            # remove NaNs
            mask = ~np.isnan(subj_x) & ~np.isnan(subj_y)
            subj_x, subj_y = subj_x[mask], subj_y[mask]

            # skip empty or constant
            if len(subj_x) < 2 or np.all(subj_x == subj_x[0]):
                continue

            try:
                coef = np.polyfit(subj_x, subj_y, 1)
                poly_fn = np.poly1d(coef)
                subj_preds.append(poly_fn(x_line))
                slopes.append(coef[0])
            except np.linalg.LinAlgError:
                continue  # skip bad fits

        if len(subj_preds) > 0:
            subj_preds = np.array(subj_preds)
            mean_pred = np.nanmean(subj_preds, axis=0)

            # 95% CI across subjects
            ci = sem(subj_preds, axis=0, nan_policy="omit") * t.ppf(0.975, df=subj_preds.shape[0]-1)

            plt.plot(x_line, mean_pred, color=colors[i], linewidth=2,
                     label=f"{group} (mean slope={np.mean(slopes):.3f}, n={len(slopes)})")
            plt.fill_between(x_line, mean_pred-ci, mean_pred+ci, color=colors[i], alpha=0.2)

    plt.xlabel('I_norm2')
    plt.ylabel('X_norm2')
    plt.title(f'I_norm2 vs X_norm2 per group {NOISE_TYPE} a{a}')
    plt.legend()
    plt.tight_layout()
    plt.show()


def figures_I_tmax_norm1_norm2(group, subject,I_tmax, I_norm1, I_norm2,a=False):
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
            for parcel in range(I_tmax_group.shape[1]):
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
        fig_name = f"box_parcel_X_norm2_a{a}_N{NPARCELLS}_{NOISE_TYPE}"
        save_path = os.path.join(FDT_parcel_subfolder, fig_name)
        p_values.plotComparisonAcrossLabels2(
            resI_Itmax,
            custom_test=statannotations_permutation.stat_permutation_test,
            columnLables=['HC', 'MCI', 'AD'],
            graphLabel=f'FDT X_norm2 Parcels {NOISE_TYPE} a{a}',
            save_path=save_path
        )
        plt.rcParams.update({'font.size': 15})
        fig_name = f"box_parcel_norm1_a{a}_N{NPARCELLS}_{NOISE_TYPE}"
        save_path = os.path.join(Inorm1_group_subfolder, fig_name)
        p_values.plotComparisonAcrossLabels2(
            resI_norm1,
            custom_test=statannotations_permutation.stat_permutation_test,
            columnLables=['HC', 'MCI', 'AD'],
            graphLabel=f'FDT I Norm1 Parcels {NOISE_TYPE} a{a}',
            save_path=save_path
        )
        plt.rcParams.update({'font.size': 15})
        fig_name = f"box_parcel_norm2_a{a}_N{NPARCELLS}_{NOISE_TYPE}"
        save_path = os.path.join(Inorm2_group_subfolder, fig_name)
        p_values.plotComparisonAcrossLabels2(
            resI_norm2,
            custom_test=statannotations_permutation.stat_permutation_test,
            columnLables=['HC', 'MCI', 'AD'],
            graphLabel=f'FDT I Norm2 Parcels {NOISE_TYPE} a{a}',
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
                if not np.isnan(I_tmax_sub_mean[groupidx, subject]):
                    records_subject_Itmax.append({
                        "value": I_tmax_sub_mean[groupidx, subject],
                        "cond": group_name,
                        "subject": subject
                    })
                if not np.isnan(I_norm1_sub_mean[groupidx, subject]):
                    records_subject_norm1.append({
                        "value": I_norm1_sub_mean[groupidx, subject],
                        "cond": group_name,
                        "subject": subject
                    })
                if not np.isnan(I_norm2_sub_mean[groupidx, subject]):
                    records_subject_norm2.append({
                        "value": I_norm2_sub_mean[groupidx, subject],
                        "cond": group_name,
                        "subject": subject
                    })

        data_subjects_Itmax = pd.DataFrame.from_records(records_subject_Itmax)
        data_subjects_norm1 = pd.DataFrame.from_records(records_subject_norm1)
        data_subjects_norm2 = pd.DataFrame.from_records(records_subject_norm2)

        fig, ax = plt.subplots(figsize=(10, 10))
        fig_name = f"violin_subject_Xnorm2_a{a}_N{NPARCELLS}_{NOISE_TYPE}"
        save_path = os.path.join(FDT_subject_subfolder, fig_name)
        plot_violins_HC_MCI_AD(
            ax=ax,
            data=data_subjects_Itmax,
            font_scale=1.4,
            metric='X_norm2 [Subject mean]',
            point_size=5,
            xgrid=False,
            plot_title=f'FDT X_norm2 — Mean per subject per group {NOISE_TYPE} a{a}',
            saveplot=1,
            filename=save_path,
            dpi=300
        )
        fig, ax = plt.subplots(figsize=(10, 10))
        fig_name = f"violin_subject_norm1_a{a}_N{NPARCELLS}_{NOISE_TYPE}"
        save_path = os.path.join(Inorm1_sub_subfolder, fig_name)
        plot_violins_HC_MCI_AD(
            ax=ax,
            data=data_subjects_norm1,
            font_scale=1.4,
            metric='I Norm1 [Subject mean]',
            point_size=5,
            xgrid=False,
            plot_title=f'FDT I Norm1 — Mean per subject per group {NOISE_TYPE} a{a}',
            saveplot=1,
            filename=save_path,
            dpi=300
        )
        fig, ax = plt.subplots(figsize=(10, 10))
        fig_name = f"violin_subject_norm2_a{a}_N{NPARCELLS}_{NOISE_TYPE}"
        save_path = os.path.join(Inorm2_sub_subfolder, fig_name)
        plot_violins_HC_MCI_AD(
            ax=ax,
            data=data_subjects_norm2,
            font_scale=1.4,
            metric='I Norm2 [Subject mean]',
            point_size=5,
            xgrid=False,
            plot_title=f'FDT I Norm2 — Mean per subject per group {NOISE_TYPE} a{a}',
            saveplot=1,
            filename=save_path,
            dpi=300
        )

def figures_barplot_parcels(option,I_tmax_group,NPARCELLS,a=False):
    if option == 'I_tmax':
        I_group = I_tmax_group
    elif option == 'I_norm1':
        I_group = I_norm1_group
    elif option == 'I_norm2':
        I_group = I_norm2_group
    else:
        raise ValueError("Invalid option. Choose from 'I_tmax', 'I_norm1', or 'I_norm2'.")

    colors = ['tab:blue', 'tab:red', 'tab:green']

    plt.figure(figsize=(12, 6))
    fig_name = f"barplot_parcel_{option}_N{NPARCELLS}_{NOISE_TYPE}_a{a}"
    save_path = os.path.join(FDT_parcel_subfolder, fig_name)
    bottom = np.zeros(NPARCELLS)  # start at zero for stacking
    for i in range(3):
        plt.bar(range(NPARCELLS), I_group[i], color=colors[i], label=f'{["HC", "MCI", "AD"][i]}', alpha=0.45)
        #bottom += I_group[i]

    plt.xlabel('Parcel')
    plt.ylabel(f'{option}')
    plt.title(f'{option} for Parcels {NOISE_TYPE} a{a}')
    plt.legend()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_means_per_RSN(name, I_tmax_group, NPARCELLS,a=False):

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
    fig_name = f"barplot_RSN_{name}_N{NPARCELLS}_{NOISE_TYPE}_a{a}"
    save_path = os.path.join(FDT_parcel_subfolder, fig_name)
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(RSNs))
    width = 0.25

    for i, group in enumerate(group_names):
        ax.bar(x + i*width - width, means_per_group[i], width, label=group)

    ax.set_xticks(x)
    ax.set_xticklabels(RSNs.keys(), rotation=45)
    ax.set_ylabel(f'Mean {name}')
    ax.set_title(f'Mean {name} per RSN {NOISE_TYPE} a{a}')
    ax.legend()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_means_per_subjects_per_RSN(RSN, I_tmax_sub, nameRSN, nameI, NPARCELLS,a=False):
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
    fig_name = f"barplot_{nameRSN}_sub_{nameI}_N{NPARCELLS}_{NOISE_TYPE}_a{a}"
    save_path = os.path.join(FDT_subject_subfolder, fig_name)
    ax.bar(range(len(means)), means, color=bar_colors)
    ax.set_xticks(range(len(means)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_ylabel(f'Mean {nameI}')
    ax.set_title(f'Mean {nameI} for {nameRSN} RSN {NOISE_TYPE} a{a}')

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

def left_right_brain_map(name,I_tmax_group,COND,NPARCELLS,a=False):
    """
    Visualizes the group differences in I(tmax, 0) on a brain map.
    """
    nii_path = os.path.join('ADNI-A_DATA', 'MNI_Glasser_HCP_v1.0.nii.gz')
    parcel_img = nib.load(nii_path)
    parcel_data = parcel_img.get_fdata()
    group_map = np.zeros_like(parcel_data)

    group_values = I_tmax_group[COND,:]

    for i in range(min(NPARCELLS,180)):
        group_map[parcel_data == i + 1] = group_values[i]
        if NPARCELLS > 180:
            group_map[parcel_data == i + 1001] = group_values[i + 180]

    group_img = nib.Nifti1Image(group_map, affine=parcel_img.affine)
    fsaverage = datasets.fetch_surf_fsaverage()

    texture_left = surface.vol_to_surf(group_img, fsaverage.pial_left)
    texture_right = surface.vol_to_surf(group_img, fsaverage.pial_right)

    vmin = np.min(group_values)
    vmax = np.max(group_values)

    fig = plt.figure(figsize=(10, 5))
    fig_name = f"left_right_brain{name}_N{NPARCELLS}_{NOISE_TYPE}_a{a}"
    save_path = os.path.join(FDT_parcel_subfolder, fig_name)

    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    plotting.plot_surf_stat_map(fsaverage.pial_left, texture_left,
                                hemi='left', title = f'{name} Left a{a}',
                                view='lateral',
                                colorbar=False, cmap='viridis',
                                bg_map=fsaverage.sulc_left,
                                vmin=0, vmax=0.6,
                                axes=ax1, darkness=None)

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    plotting.plot_surf_stat_map(fsaverage.pial_right, texture_right,
                                hemi='right', title = f'{name} Right a{a}',
                                view='lateral',
                                colorbar=False, cmap='viridis',
                                bg_map=fsaverage.sulc_right,
                                vmin=0, vmax=0.6,
                                axes=ax2, darkness=None)

    norm = Normalize(vmin=0, vmax=0.6)
    sm = ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])

    # Define position: [left, bottom, width, height] in figure coordinates (0 to 1)
    cbar_ax = fig.add_axes([0.47, 0.25, 0.02, 0.5])  # adjust as needed
    # Create the colorbar manually in that position
    cbar = plt.colorbar(sm, cax=cbar_ax)
    # cbar.set_label("Group Difference")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def brain_map_3D(name, I_tmax_group, COND, NPARCELLS,a=False):

    fsaverage = datasets.fetch_surf_fsaverage()
    nii_path = os.path.join('ADNI-A_DATA', 'MNI_Glasser_HCP_v1.0.nii.gz')
    parcel_img = nib.load(nii_path)
    parcel_data = parcel_img.get_fdata()
    group_map = np.zeros_like(parcel_data)

    group_values = I_tmax_group[COND,:]
    
    for i in range(min(NPARCELLS,180)):
        group_map[parcel_data == i + 1] = group_values[i]
        if NPARCELLS > 180:
            group_map[parcel_data == i + 1001] = group_values[i + 180]

    group_img = nib.Nifti1Image(group_map, affine=parcel_img.affine)
        
    texture_left = surface.vol_to_surf(group_img, fsaverage.pial_left)
    texture_right = surface.vol_to_surf(group_img, fsaverage.pial_right)

    surf_map = np.concatenate([texture_left, texture_right])
    coords_left, faces_left = surface.load_surf_mesh(fsaverage.pial_left)
    coords_right, faces_right = surface.load_surf_mesh(fsaverage.pial_right)

    coords = np.vstack([coords_left, coords_right])
    faces = np.vstack([faces_left, faces_right + coords_left.shape[0]])
    mesh_both = (coords, faces)

    # Plot interactively
    surf_map_masked = np.where(surf_map < 0, surf_map, np.nan)
    view = plotting.view_surf(surf_mesh=mesh_both,
                            surf_map=surf_map,
                            cmap='viridis',
                            vmin=np.nanmin(surf_map),     # Minimum value of colorbar
                            vmax=np.max(surf_map),     # Maximum value of colorbar
                            symmetric_cmap=False,
                            colorbar=True,
                            darkness=None,
                            title=f'{name}')
    view.open_in_browser()  # or just `view` if using Jupyter
    view.save_as_html(f'surface_plot_{name}_N{NPARCELLS}_{NOISE_TYPE}_a{a}.html')
    view

def load_PET_data(NPARCELLS):
    DL = ADNI_A.ADNI_A(normalizeBurden=False)
    HC_IDs = DL.get_groupSubjects('HC')
    MCI_IDs = DL.get_groupSubjects('MCI')
    AD_IDs = DL.get_groupSubjects('AD')
    HC_ABeta = []
    HC_Tau = []
    MCI_ABeta = []
    MCI_Tau = []
    AD_ABeta = []
    AD_Tau = []
    for subject in HC_IDs:
        data = DL.get_subjectData(subject,printInfo=False)
        HC_ABeta.append(np.vstack(data[subject]['ABeta'])) 
        HC_Tau.append(np.vstack(data[subject]['Tau']))
    for subject in MCI_IDs:
        data = DL.get_subjectData(subject,printInfo=False)
        MCI_ABeta.append(np.vstack(data[subject]['ABeta']))
        MCI_Tau.append(np.vstack(data[subject]['Tau']))
    for subject in AD_IDs:
        data = DL.get_subjectData(subject,printInfo=False)
        AD_ABeta.append(np.vstack(data[subject]['ABeta']))
        AD_Tau.append(np.vstack(data[subject]['Tau']))
    ABeta_burden = [np.array(HC_ABeta)[:,:NPARCELLS,0], np.array(MCI_ABeta)[:,:NPARCELLS,0], np.array(AD_ABeta)[:,:NPARCELLS,0]]
    Tau_burden = [np.array(HC_Tau)[:,:NPARCELLS,0], np.array(MCI_Tau)[:,:NPARCELLS,0], np.array(AD_Tau)[:,:NPARCELLS,0]]
    return ABeta_burden, Tau_burden

# Dictionary mapping parcel indices (1-based) to their names
Parcel_names = {
    1: "Right_V1", 2: "Right_MST", 3: "Right_V6", 4: "Right_V2", 5: "Right_V3", 6: "Right_V4", 7: "Right_V8",
    8: "Right_4", 9: "Right_3b", 10: "Right_FEF", 11: "Right_PEF", 12: "Right_55b", 13: "Right_V3A", 14: "Right_RSC",
    15: "Right_POS2", 16: "Right_V7", 17: "Right_IPS1", 18: "Right_FFC", 19: "Right_V3B", 20: "Right_LO1",
    21: "Right_LO2", 22: "Right_PIT", 23: "Right_MT", 24: "Right_A1", 25: "Right_PSL", 26: "Right_SFL",
    27: "Right_PCV", 28: "Right_STV", 29: "Right_7Pm", 30: "Right_7m", 31: "Right_POS1", 32: "Right_23d",
    33: "Right_v23ab", 34: "Right_d23ab", 35: "Right_31pv", 36: "Right_5m", 37: "Right_5mv", 38: "Right_23c",
    39: "Right_5L", 40: "Right_24dd", 41: "Right_24dv", 42: "Right_7AL", 43: "Right_SCEF", 44: "Right_6ma",
    45: "Right_7Am", 46: "Right_7PL", 47: "Right_7PC", 48: "Right_LIPv", 49: "Right_VIP", 50: "Right_MIP",
    51: "Right_1", 52: "Right_2", 53: "Right_3a", 54: "Right_6d", 55: "Right_6mp", 56: "Right_6v",
    57: "Right_p24pr", 58: "Right_33pr", 59: "Right_a24pr", 60: "Right_p32pr", 61: "Right_a24", 62: "Right_d32",
    63: "Right_8BM", 64: "Right_p32", 65: "Right_10r", 66: "Right_47m", 67: "Right_8Av", 68: "Right_8Ad",
    69: "Right_9m", 70: "Right_8BL", 71: "Right_9p", 72: "Right_10d", 73: "Right_8C", 74: "Right_44",
    75: "Right_45", 76: "Right_47l", 77: "Right_a47r", 78: "Right_6r", 79: "Right_IFJa", 80: "Right_IFJp",
    81: "Right_IFSp", 82: "Right_IFSa", 83: "Right_p9-46v", 84: "Right_46", 85: "Right_a9-46v", 86: "Right_9-46d",
    87: "Right_9a", 88: "Right_10v", 89: "Right_a10p", 90: "Right_10pp", 91: "Right_11l", 92: "Right_13l",
    93: "Right_OFC", 94: "Right_47s", 95: "Right_LIPd", 96: "Right_6a", 97: "Right_i6-8", 98: "Right_s6-8",
    99: "Right_43", 100: "Right_OP4", 101: "Right_OP1", 102: "Right_OP2-3", 103: "Right_52", 104: "Right_RI",
    105: "Right_PFcm", 106: "Right_PoI2", 107: "Right_TA2", 108: "Right_FOP4", 109: "Right_MI", 110: "Right_Pir",
    111: "Right_AVI", 112: "Right_AAIC", 113: "Right_FOP1", 114: "Right_FOP3", 115: "Right_FOP2", 116: "Right_PFt",
    117: "Right_AIP", 118: "Right_EC", 119: "Right_PreS", 120: "Right_H", 121: "Right_ProS", 122: "Right_PeEc",
    123: "Right_STGa", 124: "Right_PBelt", 125: "Right_A5", 126: "Right_PHA1", 127: "Right_PHA3", 128: "Right_STSda",
    129: "Right_STSdp", 130: "Right_STSvp", 131: "Right_TGd", 132: "Right_TE1a", 133: "Right_TE1p", 134: "Right_TE2a",
    135: "Right_TF", 136: "Right_TE2p", 137: "Right_PHT", 138: "Right_PH", 139: "Right_TPOJ1", 140: "Right_TPOJ2",
    141: "Right_TPOJ3", 142: "Right_DVT", 143: "Right_PGp", 144: "Right_IP2", 145: "Right_IP1", 146: "Right_IP0",
    147: "Right_PFop", 148: "Right_PF", 149: "Right_PFm", 150: "Right_PGi", 151: "Right_PGs", 152: "Right_V6A",
    153: "Right_VMV1", 154: "Right_VMV3", 155: "Right_PHA2", 156: "Right_V4t", 157: "Right_FST", 158: "Right_V3CD",
    159: "Right_LO3", 160: "Right_VMV2", 161: "Right_31pd", 162: "Right_31a", 163: "Right_VVC", 164: "Right_25",
    165: "Right_s32", 166: "Right_pOFC", 167: "Right_PoI1", 168: "Right_Ig", 169: "Right_FOP5", 170: "Right_p10p",
    171: "Right_p47r", 172: "Right_TGv", 173: "Right_MBelt", 174: "Right_LBelt", 175: "Right_A4", 176: "Right_STSva",
    177: "Right_TE1m", 178: "Right_PI", 179: "Right_a32pr", 180: "Right_p24",
    181: "Left_V1", 182: "Left_MST", 183: "Left_V6", 184: "Left_V2", 185: "Left_V3", 186: "Left_V4", 187: "Left_V8",
    188: "Left_4", 189: "Left_3b", 190: "Left_FEF", 191: "Left_PEF", 192: "Left_55b", 193: "Left_V3A", 194: "Left_RSC",
    195: "Left_POS2", 196: "Left_V7", 197: "Left_IPS1", 198: "Left_FFC", 199: "Left_V3B", 200: "Left_LO1",
    201: "Left_LO2", 202: "Left_PIT", 203: "Left_MT", 204: "Left_A1", 205: "Left_PSL", 206: "Left_SFL",
    207: "Left_PCV", 208: "Left_STV", 209: "Left_7Pm", 210: "Left_7m", 211: "Left_POS1", 212: "Left_23d",
    213: "Left_v23ab", 214: "Left_d23ab", 215: "Left_31pv", 216: "Left_5m", 217: "Left_5mv", 218: "Left_23c",
    219: "Left_5L", 220: "Left_24dd", 221: "Left_24dv", 222: "Left_7AL", 223: "Left_SCEF", 224: "Left_6ma",
    225: "Left_7Am", 226: "Left_7PL", 227: "Left_7PC", 228: "Left_LIPv", 229: "Left_VIP", 230: "Left_MIP",
    231: "Left_1", 232: "Left_2", 233: "Left_3a", 234: "Left_6d", 235: "Left_6mp", 236: "Left_6v",
    237: "Left_p24pr", 238: "Left_33pr", 239: "Left_a24pr", 240: "Left_p32pr", 241: "Left_a24", 242: "Left_d32",
    243: "Left_8BM", 244: "Left_p32", 245: "Left_10r", 246: "Left_47m", 247: "Left_8Av", 248: "Left_8Ad",
    249: "Left_9m", 250: "Left_8BL", 251: "Left_9p", 252: "Left_10d", 253: "Left_8C", 254: "Left_44",
    255: "Left_45", 256: "Left_47l", 257: "Left_a47r", 258: "Left_6r", 259: "Left_IFJa", 260: "Left_IFJp",
    261: "Left_IFSp", 262: "Left_IFSa", 263: "Left_p9-46v", 264: "Left_46", 265: "Left_a9-46v", 266: "Left_9-46d",
    267: "Left_9a", 268: "Left_10v", 269: "Left_a10p", 270: "Left_10pp", 271: "Left_11l", 272: "Left_13l",
    273: "Left_OFC", 274: "Left_47s", 275: "Left_LIPd", 276: "Left_6a", 277: "Left_i6-8", 278: "Left_s6-8",
    279: "Left_43", 280: "Left_OP4", 281: "Left_OP1", 282: "Left_OP2-3", 283: "Left_52", 284: "Left_RI",
    285: "Left_PFcm", 286: "Left_PoI2", 287: "Left_TA2", 288: "Left_FOP4", 289: "Left_MI", 290: "Left_Pir",
    291: "Left_AVI", 292: "Left_AAIC", 293: "Left_FOP1", 294: "Left_FOP3", 295: "Left_FOP2", 296: "Left_PFt",
    297: "Left_AIP", 298: "Left_EC", 299: "Left_PreS", 300: "Left_H", 301: "Left_ProS", 302: "Left_PeEc",
    303: "Left_STGa", 304: "Left_PBelt", 305: "Left_A5", 306: "Left_PHA1", 307: "Left_PHA3", 308: "Left_STSda",
    309: "Left_STSdp", 310: "Left_STSvp", 311: "Left_TGd", 312: "Left_TE1a", 313: "Left_TE1p", 314: "Left_TE2a",
    315: "Left_TF", 316: "Left_TE2p", 317: "Left_PHT", 318: "Left_PH", 319: "Left_TPOJ1", 320: "Left_TPOJ2",
    321: "Left_TPOJ3", 322: "Left_DVT", 323: "Left_PGp", 324: "Left_IP2", 325: "Left_IP1", 326: "Left_IP0",
    327: "Left_PFop", 328: "Left_PF", 329: "Left_PFm", 330: "Left_PGi", 331: "Left_PGs", 332: "Left_V6A",
    333: "Left_VMV1", 334: "Left_VMV3", 335: "Left_PHA2", 336: "Left_V4t", 337: "Left_FST", 338: "Left_V3CD",
    339: "Left_LO3", 340: "Left_VMV2", 341: "Left_31pd", 342: "Left_31a", 343: "Left_VVC", 344: "Left_25",
    345: "Left_s32", 346: "Left_pOFC", 347: "Left_PoI1", 348: "Left_Ig", 349: "Left_FOP5", 350: "Left_p10p",
    351: "Left_p47r", 352: "Left_TGv", 353: "Left_MBelt", 354: "Left_LBelt", 355: "Left_A4", 356: "Left_STSva",
    357: "Left_TE1m", 358: "Left_PI", 359: "Left_a32pr", 360: "Left_p24", 361: "Right_Thalamus", 362: "Right_Caudate",
    363: "Right_Putamen", 364: "Right_Pallidum", 365: "Right_Hippocampus",
    366: "Right_Amygdala", 367: "Right_Nucleus accumbens", 368: "Right_Ventral diencephalon", 369: "Right_Cerebellar cortex",
    370: "Left_Thalamus", 371: "Left_Caudate", 372: "Left_Putamen", 373: "Left_Pallidum", 374: "Left_Hippocampus", 375: "Left_Amygdala",
    376: "Left_Nucleus accumbens", 377: "Left_Ventral diencephalon", 378: "Left_Cerebellar cortex", 379: "Brainstem"
}
####################################################################

NPARCELLS = 379
NOISE_TYPE = 'hetero'#"HOMO"
A_FITTING = True
all_values = None
all_values_a = None
if A_FITTING:
    all_values_a = load_appended_records(
        filepath=os.path.join(FDT_values_subfolder, f"FDT_values_a{A_FITTING}_N{NPARCELLS}_{NOISE_TYPE}.npz")
    )
    I_tmax_group_a = np.squeeze(np.array(get_field(all_values_a, "I_tmax", filters={"level": "group"})), axis=0)
    I_norm1_group_a = np.squeeze(np.array(get_field(all_values_a, "I_norm1", filters={"level": "group"})), axis=0)
    I_norm2_group_a = np.squeeze(np.array(get_field(all_values_a, "I_norm2", filters={"level": "group"})), axis=0)
    X_norm2_group_a = np.squeeze(np.array(get_field(all_values_a, "X_Inorm2", filters={"level": "group"})), axis=0)
    I_tmax_sub_a = np.squeeze(np.array(get_field(all_values_a, "I_tmax", filters={"level": "subject"})), axis=0)
    I_norm1_sub_a = np.squeeze(np.array(get_field(all_values_a, "I_norm1", filters={"level": "subject"})), axis=0)
    I_norm2_sub_a = np.squeeze(np.array(get_field(all_values_a, "I_norm2", filters={"level": "subject"})), axis=0)
    X_norm2_sub_a = np.squeeze(np.array(get_field(all_values_a, "X_Inorm2", filters={"level": "subject"})), axis=0)
    a_values_group = np.squeeze(get_field(all_values_a, "a", filters={"level": "group"}))
    a_values_sub = np.split(get_field(all_values_a, "a", filters={"level": "subject"})[0], [17, 26], axis=0)
    a_original_group = np.squeeze(np.array(get_field(all_values_a, "original_a", filters={"level": "group"})))
    a_original_sub = get_field(all_values_a, "original_a", filters={"level": "subject"})[0][0]

all_values = load_appended_records(
    filepath=os.path.join(FDT_values_subfolder, f"FDT_values_a{A_FITTING}_N{NPARCELLS}_{NOISE_TYPE}.npz")
)
I_tmax_group = np.squeeze(np.array(get_field(all_values, "I_tmax", filters={"level": "group"})), axis=0)
I_norm1_group = np.squeeze(np.array(get_field(all_values, "I_norm1", filters={"level": "group"})), axis=0)
I_norm2_group = np.squeeze(np.array(get_field(all_values, "I_norm2", filters={"level": "group"})), axis=0)
X_norm2_group = np.squeeze(np.array(get_field(all_values, "X_Inorm2", filters={"level": "group"})), axis=0)
I_tmax_sub = np.squeeze(np.array(get_field(all_values, "I_tmax", filters={"level": "subject"})), axis=0)
I_norm1_sub = np.squeeze(np.array(get_field(all_values, "I_norm1", filters={"level": "subject"})), axis=0)
I_norm2_sub = np.squeeze(np.array(get_field(all_values, "I_norm2", filters={"level": "subject"})), axis=0)
X_norm2_sub = np.squeeze(np.array(get_field(all_values, "X_Inorm2", filters={"level": "subject"})), axis=0)
ABeta_burden, Tau_burden = load_PET_data(min(NPARCELLS,360))

if A_FITTING:
    a_002 = -0.02 * np.ones(a_values_group.shape)
    diff_a_group = np.subtract(a_values_group, a_original_group)
    diff_a_sub = np.subtract(a_values_sub[0], a_original_sub[0])
    diff_I_norm1_a_subHC = np.subtract(I_norm1_sub_a[0], I_norm1_sub[0])
    diff_org_a_group = np.subtract(a_values_group, a_002)

 
I_norm2_select = np.array([I_norm2_sub[0,0,:], I_norm2_sub[1,0,:], I_norm2_sub[2,0,:]])
X_norm2_select = np.array([X_norm2_sub[0,0,:], X_norm2_sub[1,0,:], X_norm2_sub[2,0,:]])

plot_boxes_HC_MCI_AD(data=ABeta_burden,saveplot=1,metric='ABeta burden',plot_title='Abeta burden subject-average across groups',filename=f'Abeta_burden_N{NPARCELLS}_{NOISE_TYPE}_a{A_FITTING}')
plot_boxes_HC_MCI_AD(data=Tau_burden,saveplot=1,metric='Tau burden',plot_title='Tau burden subject-average across groups',filename=f'Tau_burden_N{NPARCELLS}_{NOISE_TYPE}_a{A_FITTING}')
# I_vs_Xnorm2(I_norm2_group_a, X_norm2_group_a, a=True)
# I_vs_Xnorm2(I_norm2_sub_a, X_norm2_sub_a, a=True, sub=True)
# if A_FITTING: I_vs_Xnorm2(I_norm2_group, X_norm2_group,I_norm2_group_a,X_norm2_group_a, a=True)
# if A_FITTING: I_vs_Xnorm2(I_norm2_sub, X_norm2_sub,I_norm2_sub_a,X_norm2_sub_a, a=True, sub=True)
# I_vs_Xnorm2(I_norm2_select, X_norm2_select, a=False)
# print(I_norm2_select.shape, I_norm2_group.shape)
# I_vs_Xnorm22(I_norm2_sub, X_norm2_sub, a=False)

# figures_I_tmax_norm1_norm2(group=True, subject=False, I_tmax=X_norm2_group, I_norm1=I_norm1_group, I_norm2=I_norm2_group)
# if A_FITTING: figures_I_tmax_norm1_norm2(group=True, subject=False, I_tmax=X_norm2_group_a, I_norm1=I_norm1_group_a, I_norm2=I_norm2_group_a,a=A_FITTING)
# # figures_I_tmax_norm1_norm2(group=False, subject=True, I_tmax=X_norm2_sub, I_norm1=I_norm1_sub, I_norm2=I_norm2_sub)
# if A_FITTING: figures_I_tmax_norm1_norm2(group=False, subject=True, I_tmax=X_norm2_sub_a, I_norm1=I_norm1_sub_a, I_norm2=I_norm2_sub_a, a=A_FITTING)

# figures_barplot_parcels('I_tmax',I_tmax_group, NPARCELLS)
# if A_FITTING: figures_barplot_parcels('I_tmax', I_tmax_group_a, NPARCELLS, a=A_FITTING)
# figures_barplot_parcels('I_norm1', I_norm1_group, NPARCELLS)
# if A_FITTING: figures_barplot_parcels('I_norm1', I_norm1_group_a, NPARCELLS, a=A_FITTING)
# figures_barplot_parcels('I_norm2', I_norm2_group, NPARCELLS)
# if A_FITTING: figures_barplot_parcels('I_norm2', I_norm2_group_a, NPARCELLS, a=A_FITTING)


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


# if A_FITTING: 
#     RSN_significance_group(I_norm2_group_a, a=A_FITTING)
#     RSN_radar_plot(I_norm2_group_a, a=A_FITTING)
# RSN_significance_group(I_norm2_group, a=False)
# RSN_radar_plot(I_norm2_group, a=False)
# plot_means_per_RSN('I_tmax', I_tmax_group, NPARCELLS)
# if A_FITTING: plot_means_per_RSN('I_tmax_a', I_tmax_group_a, NPARCELLS,a=A_FITTING)
# plot_means_per_RSN('I_norm1', I_norm1_group, NPARCELLS)
# if A_FITTING: plot_means_per_RSN('I_norm1_a', I_norm1_group_a, NPARCELLS,a=A_FITTING)
# plot_means_per_RSN('I_norm2', I_norm2_group, NPARCELLS)
# if A_FITTING: plot_means_per_RSN('I_norm2_a', I_norm2_group_a, NPARCELLS,a=A_FITTING)

# plot_means_per_subjects_per_RSN(SomMot, I_tmax_sub, 'SomMot', 'I_tmax', NPARCELLS)
# if A_FITTING: plot_means_per_subjects_per_RSN(SomMot, I_tmax_sub_a, 'SomMot', 'I_tmax', NPARCELLS,a=A_FITTING)
# plot_means_per_subjects_per_RSN(Vis, I_tmax_sub, 'Vis', 'I_tmax', NPARCELLS)
# if A_FITTING: plot_means_per_subjects_per_RSN(Vis, I_tmax_sub_a, 'Vis', 'I_tmax', NPARCELLS,a=A_FITTING)
# plot_means_per_subjects_per_RSN(Limbic, I_tmax_sub, 'Limbic', 'I_tmax', NPARCELLS)
# if A_FITTING: plot_means_per_subjects_per_RSN(Limbic, I_tmax_sub_a, 'Limbic', 'I_tmax', NPARCELLS,a=A_FITTING)
# #...

###### VISUALIZATION ######
# left_right_brain_map('I_Norm2_HC', I_norm2_group, 0, NPARCELLS)
# left_right_brain_map('I_Norm2_MCI', I_norm2_group, 1, NPARCELLS)
# left_right_brain_map('I_Norm2_AD', I_norm2_group, 2, NPARCELLS)

# brain_map_3D(f'I_Norm2_HC_{NOISE_TYPE}', I_norm2_group, 0, NPARCELLS)
# if A_FITTING: brain_map_3D(f'I_tmax_HC_{NOISE_TYPE}_a{A_FITTING}', I_tmax_group_a, 0, NPARCELLS, a=A_FITTING)
# brain_map_3D(f'I_Norm2_MCI_{NOISE_TYPE}', I_norm2_group, 1, NPARCELLS)
# if A_FITTING: brain_map_3D(f'I_tmax_MCI_{NOISE_TYPE}_a{A_FITTING}', I_tmax_group_a, 1, NPARCELLS, a=A_FITTING)
# brain_map_3D(f'I_Norm2_AD_{NOISE_TYPE}', I_norm2_group, 2, NPARCELLS)
# if A_FITTING: brain_map_3D(f'I_tmax_AD_{NOISE_TYPE}_a{A_FITTING}', I_tmax_group_a, 2, NPARCELLS, a=A_FITTING)
# brain_map_3D(f'I_tmax_HC_{NOISE_TYPE}_0', I_tmax_sub[0], 0, NPARCELLS)
# brain_map_3D(f'I_tmax_HC_{NOISE_TYPE}_1', I_tmax_sub[0], 1, NPARCELLS)
# brain_map_3D(f'I_tmax_HC_{NOISE_TYPE}_2', I_tmax_sub[0], 2, NPARCELLS)
# brain_map_3D(f'I_tmax_HC_{NOISE_TYPE}_3', I_tmax_sub[0], 3, NPARCELLS)
# brain_map_3D(f'I_tmax_HC_{NOISE_TYPE}_4', I_tmax_sub[0], 4, NPARCELLS)
# brain_map_3D(f'I_tmax_HC_{NOISE_TYPE}_5', I_tmax_sub[0], 5, NPARCELLS)

# brain_map_3D(f'a_original_group_HC_{NOISE_TYPE}', diff_org_a_group, 0, NPARCELLS)
# brain_map_3D(f'a_original_group_MCI_{NOISE_TYPE}', diff_org_a_group, 1, NPARCELLS)
# brain_map_3D(f'a_original_group_AD_{NOISE_TYPE}', diff_org_a_group, 2, NPARCELLS)

# brain_map_3D(f'diff_a_sub_HC_{NOISE_TYPE}', diff_a_sub, 0, NPARCELLS)
# brain_map_3D(f'a_sub_HC_{NOISE_TYPE}', a_values_sub[0], 0, NPARCELLS)
# brain_map_3D(f'I_norm1_a_sub_HC_{NOISE_TYPE}', I_norm1_sub_a[0], 0, NPARCELLS)
# brain_map_3D(f'I_norm1_a_sub_MCI_{NOISE_TYPE}', I_norm1_sub_a[1], 0, NPARCELLS)
# brain_map_3D(f'I_norm1_a_sub_AD_{NOISE_TYPE}', I_norm1_sub_a[2], 0, NPARCELLS)
# brain_map_3D(f'I_norm1_sub_HC_{NOISE_TYPE}', I_norm1_sub[0], 0, NPARCELLS)
# brain_map_3D(f'diff_I_norm1_sub1_{NOISE_TYPE}', diff_I_norm1_a_subHC, 0, NPARCELLS)

groups = ["HC", "MCI", "AD"]
colors = ["tab:blue", "tab:orange", "tab:green"]

# 1. Compute range per parcel
if A_FITTING: 
    diffs = np.max(I_norm2_group_a, axis=0) - np.min(I_norm2_group_a, axis=0)

    # 2. Find top N
    top_n = 18
    top_parcels_nonsort = np.argsort(diffs)[::-1][:top_n]
    top_parcels = np.sort(top_parcels_nonsort)  # sort indices for plotting
    # 3. Prepare bar plot
    x = np.arange(len(top_parcels))  # parcel positions

    fig, ax = plt.subplots(figsize=(12, 6))
    bar_width = 0.25

    for i, group in enumerate(groups):
        ax.bar(
            x + i * bar_width,
            I_norm2_group_a[i, top_parcels],
            width=bar_width,
            label=group,
            color=colors[i]
        )

    # 4. Set labels
    # print([Parcel_names.get(idx+1, f"Parcel {idx+1}") for idx in top_parcels])
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels([Parcel_names.get(idx+1, f"Parcel {idx+1}") for idx in top_parcels], rotation=45, ha="right")
    ax.set_ylabel("I_tmax")
    ax.set_title("I_tmax HOMO top parcels with largest between-group differences")
    ax.legend()

    plt.tight_layout()
    # plt.show()

    top6_parcels = top_parcels_nonsort[:6]  # first 6 parcels for detailed analysis
    x = np.arange(len(top6_parcels)*27)  # parcel positions

    n_parcels = len(top6_parcels)
    n_groups = len(groups)
    n_subjects = 4

    bar_width = 0.2
    x = np.arange(n_parcels)  # positions for parcels

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, group in enumerate(groups):
        # We plot **all subjects in that group** with a small offset for clarity
        for subj in range(n_subjects):
            ax.bar(
                x + i * bar_width + subj*0.02,  # small shift per subject
                I_norm2_sub_a[i, subj, top6_parcels],
                width=0.02,
                color=colors[i],
                alpha=0.7
            )

    # Set labels
    parcel_labels = [Parcel_names.get(idx+1, f"Parcel {idx+1}") for idx in top6_parcels]
    ax.set_xticks(x + bar_width)  # center ticks
    ax.set_xticklabels(parcel_labels, rotation=45, ha="right")

    ax.set_ylabel("I(tmax)")
    ax.set_title("Top 6 parcels — values per subject per group")
    ax.legend(groups)

    plt.tight_layout()
    #plt.show()


# df_cohort["ABeta_mean"] = df_cohort.groupby("subject")["ABeta_local"].transform("mean")
# df_cohort["Tau_mean"]   = df_cohort.groupby("subject")["Tau_local"].transform("mean")
# df_cohort["I_mean"]     = df_cohort.groupby("subject")["I_local"].transform("mean")
# df_cohort["X_mean"]     = df_cohort.groupby("subject")["X_local"].transform("mean")

# # Difference between mean and local value
# df_cohort["ABeta_dif"] = df_cohort["ABeta_mean"] - df_cohort["ABeta_local"]
# df_cohort["Tau_dif"]   = df_cohort["Tau_mean"] - df_cohort["Tau_local"]
# df_cohort["I_dif"]     = df_cohort["I_mean"] - df_cohort["I_local"]
# df_cohort["X_dif"]     = df_cohort["X_mean"] - df_cohort["X_local"]



from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from collections import Counter, defaultdict
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from collections import defaultdict, Counter
from sklearn.decomposition import PCA, FastICA, SparsePCA


from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.model_selection import LeaveOneOut
rng = np.random.default_rng(42)


def subject_pca_features(df, feature, n_components=5):
    """
    Extract first n_components of parcel-wise distribution of 'feature' across subjects.
    Returns dataframe: subject × components
    """
    # pivot: rows=subject, cols=parcel values of the feature
    mat = df.pivot(index="subject", columns="parcel", values=feature).to_numpy()

    alpha = 1         # sparsity parameter (larger = sparser)

    spca = SparsePCA(n_components=n_components, alpha=alpha, random_state=42)
    X_spca = spca.fit_transform(X)  # subject × component scores
    components = spca.components_   # component × (parcel × feature) loadings

    
    pca = PCA(n_components=n_components, random_state=42)
    comps = pca.fit_transform(mat)  # shape (n_subjects, n_components)
    
    return pd.DataFrame(
        comps,
        index=df["subject"].unique(),
        columns=[f"{feature}_PC{i+1}" for i in range(n_components)]
    )

def subject_ica_features(df_long, measure_col, n_components=20):
    """
    Pivots the long dataframe to (subjects x parcels) and runs ICA.
    The transform gives the component scores (the "mixing matrix") for each subject.
    Returns a wide DataFrame (subjects x components).
    """
    # Pivot to get the right shape for ICA: (n_subjects, n_features)
    data_matrix = df_long.pivot(index='subject', columns='parcel', values=measure_col)
    
    # Initialize and run ICA
    ica = FastICA(n_components=n_components, random_state=42, max_iter=1000, whiten='unit-variance')
    
    # The transform gives us the component scores for each subject
    subject_component_scores = ica.fit_transform(data_matrix)

    spatial_weights = ica.components_

    

    # Store these new features in a DataFrame
    return pd.DataFrame(
        subject_component_scores, 
        index=data_matrix.index, # Use subject IDs as index
        columns=[f'{measure_col}_IC_{i}' for i in range(n_components)]
    ), spatial_weights
   


def subject_cross_correlation(df, f1, f2):
    corrs = {}
    for subj, sub_df in df.groupby("subject"):
        corrs[subj] = np.corrcoef(sub_df[f1], sub_df[f2])[0,1]
    return pd.DataFrame.from_dict(corrs, orient="index", columns=[f"{f1}_vs_{f2}_corr"])


# --- 1. Build parcel-level dataframe ---
df_list = []

for cohort_idx, (AB, Tau, I, X) in enumerate(
        zip(ABeta_burden,
            Tau_burden,
            I_norm2_sub_a[:, :, 0:min(NPARCELLS, 360)],
            X_norm2_sub_a[:, :, 0:min(NPARCELLS, 360)])):

    nsub, nparcel = AB.shape
    I = I[:nsub, :]
    X = X[:nsub, :]

    df = pd.DataFrame({
        "subject": np.repeat([f"C{cohort_idx}_S{i}" for i in range(nsub)], nparcel),
        "parcel": np.tile(np.arange(nparcel), nsub),
        "ABeta_local": AB.ravel(),
        "Tau_local": Tau.ravel(),
        "I_local": I.ravel(),
        "X_local": X.ravel(),
        "cohort": cohort_idx
    })
    df_list.append(df)

df_cohort = pd.concat(df_list, ignore_index=True)

df_pca_AB = subject_pca_features(df_cohort, "ABeta_local")
df_pca_Tau = subject_pca_features(df_cohort, "Tau_local")
df_pca_I = subject_pca_features(df_cohort, "I_local")
df_pca_X = subject_pca_features(df_cohort, "X_local")
df_ica_AB,comAB = subject_ica_features(df_cohort, "ABeta_local", n_components=20)
df_ica_Tau,comT = subject_ica_features(df_cohort, "Tau_local", n_components=20)
df_ica_I,comI = subject_ica_features(df_cohort, "I_local", n_components=20)
df_ica_X,comX = subject_ica_features(df_cohort, "X_local", n_components=20)


print(np.sort(comAB[0])[:15])
df_corr_AB_Tau = subject_cross_correlation(df_cohort, "ABeta_local", "Tau_local")
df_corr_AB_I   = subject_cross_correlation(df_cohort, "ABeta_local", "I_local")
df_corr_AB_X   = subject_cross_correlation(df_cohort, "ABeta_local", "X_local")
df_corr_Tau_I  = subject_cross_correlation(df_cohort, "Tau_local", "I_local")
df_corr_Tau_X  = subject_cross_correlation(df_cohort, "Tau_local", "X_local")

#--- 2. Compute subject-level basic stats ---
df_stats = (
    df_cohort.groupby("subject")
    .agg(
        {f: ["mean"] for f in ["ABeta_local", "Tau_local", "I_local", "X_local"]} |
        {"cohort": "first"}  # <- add cohort here
    )
)

df_stats.columns = ["_".join(c) if c[0] != "cohort" else "cohort" for c in df_stats.columns]
df_stats = df_stats.reset_index()

# --- 3. Merge PCA features ---
df_subject_features = df_stats.copy()
# df_subject_features = df_subject_features.merge(df_pca_AB, left_on="subject", right_index=True)
# df_subject_features = df_subject_features.merge(df_pca_Tau, left_on="subject", right_index=True)
# df_subject_features = df_subject_features.merge(df_pca_I, left_on="subject", right_index=True)
# df_subject_features = df_subject_features.merge(df_pca_X, left_on="subject", right_index=True)
df_subject_features = df_subject_features.merge(df_ica_AB, left_on="subject", right_index=True)
df_subject_features = df_subject_features.merge(df_ica_Tau, left_on="subject", right_index=True)
df_subject_features = df_subject_features.merge(df_ica_I, left_on="subject", right_index=True)
df_subject_features = df_subject_features.merge(df_ica_X, left_on="subject", right_index=True)

# --- 4. Merge inter-feature correlations ---
# for df_corr in [df_corr_AB_Tau, df_corr_AB_I, df_corr_AB_X, df_corr_Tau_I, df_corr_Tau_X]:
#     df_subject_features = df_subject_features.merge(df_corr, left_on="subject", right_index=True)

# --- Final dataframe ---
#pd.set_option("display.max_rows", None)
#print(df_subject_features)

# ------------------------------------------------------------------
# EXPECTED INPUT
# One long dataframe where each row = a parcel from a subject.
# Columns (example): ['subject_id','group','parcel_idx','ABeta','Tau','I_N2','X_N2',
#                     'ABeta_mean','Tau_mean','I_N2_mean','X_N2_mean']
# You can choose any subset of feature columns below.
# ------------------------------------------------------------------

def run_subjectwise_svm(
    df,
    feature_cols=('parcel_idx','ABeta','Tau','I_N2','X_N2',   # base 5
                  'ABeta_dif','Tau_dif','I_N2_dif','X_N2_dif'),  # +4 optional
    label_col='cohort',
    group_col='subject',
    n_test_subjects=4,
    n_repeats=10,
    inner_cv_splits=5,
    kernel='linear',   # 'linear' or 'rbf'
    Cs=(0.1, 1, 10),
    gamma_vals=('scale',)  # Only used if kernel='rbf'
):
    # Make sure only the selected features exist
    feature_cols = [c for c in feature_cols if c in df.columns]
    X_all = df[feature_cols].to_numpy()
    y_all = df[label_col].to_numpy()
    groups_all = df[group_col].to_numpy()

    # List of unique subjects and their labels
    subjects = df[[group_col, label_col]].drop_duplicates()
    subj_ids = subjects[group_col].to_numpy()
    subj_labels = subjects[label_col].to_numpy()

    # Helper: sample exactly n_test_subjects, roughly stratified by label
    def sample_test_subjects():
        test_ids = []
        by_class = {}
        for g in np.unique(subj_labels):
            by_class[g] = subj_ids[subj_labels == g]
        # proportional target per class (rounded)
        total = len(subj_ids)
        target_per_class = {g: int(round(len(by_class[g]) * n_test_subjects / total))
                            for g in by_class}
        # Fix rounding to sum exactly n_test_subjects
        diff = n_test_subjects - sum(target_per_class.values())
        # Adjust classes with largest remainder (simple heuristic)
        remainders = {g: (len(by_class[g]) * n_test_subjects / total) - target_per_class[g]
                      for g in by_class}
        for g in sorted(remainders, key=remainders.get, reverse=True)[:abs(diff)]:
            target_per_class[g] += np.sign(diff)

        for g in by_class:
            k = max(0, min(target_per_class[g], len(by_class[g])))
            pick = rng.choice(by_class[g], size=k, replace=False)
            test_ids.extend(pick.tolist())

        # If still off due to edge cases, pad randomly from remaining
        if len(test_ids) < n_test_subjects:
            remaining = [s for s in subj_ids if s not in test_ids]
            extra = rng.choice(remaining, size=n_test_subjects - len(test_ids), replace=False)
            test_ids.extend(extra.tolist())
        elif len(test_ids) > n_test_subjects:
            test_ids = rng.choice(test_ids, size=n_test_subjects, replace=False).tolist()
        return set(test_ids)

    # Build model + inner CV
    param_grid = {'svc__C': Cs}
    if kernel == 'rbf':
        param_grid['svc__gamma'] = gamma_vals

    out = {
        'parcel_acc': [],
        'subject_acc': [],
        'reports': [],
        'confusions': []
    }

    for rep in range(n_repeats):
        test_subjects = sample_test_subjects()
        is_test = df[group_col].isin(test_subjects).to_numpy()
        X_train, y_train, groups_train = X_all[~is_test], y_all[~is_test], groups_all[~is_test]
        X_test,  y_test,  groups_test  = X_all[is_test],  y_all[is_test],  groups_all[is_test]

        # Inner group-aware CV (keeps subjects separate)
        inner_cv = GroupKFold(n_splits=min(inner_cv_splits, len(np.unique(groups_train))))
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('svc', SVC(kernel=kernel, class_weight='balanced', probability=False))#LinearSVC(class_weight="balanced", max_iter=5000))
        ])
        clf = GridSearchCV(
            pipe,
            param_grid=param_grid,
            cv=inner_cv.split(X_train, y_train, groups_train),
            scoring='accuracy',
            n_jobs=-1
        )
        clf.fit(X_train, y_train)

        # Parcel-level predictions
        y_pred = clf.predict(X_test)
        parcel_acc = accuracy_score(y_test, y_pred)
        out['parcel_acc'].append(parcel_acc)

        # Subject-level majority vote
        subj_preds = defaultdict(list)
        subj_trues = {}
        for yi, yp, sid in zip(y_test, y_pred, groups_test):
            subj_preds[sid].append(yp)
            subj_trues[sid] = yi
        y_true_subj = []
        y_pred_subj = []
        for sid, preds in subj_preds.items():
            maj = Counter(preds).most_common(1)[0][0]
            y_pred_subj.append(maj)
            y_true_subj.append(subj_trues[sid])
        subject_acc = accuracy_score(y_true_subj, y_pred_subj)
        out['subject_acc'].append(subject_acc)

        # Optional: keep per-run diagnostics
        out['reports'].append(classification_report(y_test, y_pred, output_dict=True))
        out['confusions'].append(confusion_matrix(y_test, y_pred, labels=np.unique(y_all)))

    # Summary
    def mean_std(x): 
        return float(np.mean(x)), float(np.std(x))
    parcel_mean, parcel_std = mean_std(out['parcel_acc'])
    subject_mean, subject_std = mean_std(out['subject_acc'])

    print(f"Parcel-level accuracy over {n_repeats} runs (holdout {n_test_subjects} subjects): "
          f"{parcel_mean:.3f} ± {parcel_std:.3f}")
    print(f"Subject-level accuracy over {n_repeats} runs (majority vote): "
          f"{subject_mean:.3f} ± {subject_std:.3f}")
    print("Example best params (last run):", clf.best_params_)

    return out

# results = run_subjectwise_svm(
#     df_cohort,
#     feature_cols=('parcel','ABeta_local','Tau_local','I_local','X_local',
#                   'ABeta_dif','Tau_dif','I_dif','X_dif'),
#     n_test_subjects=4,
#     n_repeats=50,      # increase for tighter CIs
#     kernel='rbf',    # start linear; try 'rbf' after
#     Cs=(0.1, 1, 10)
# )

feature_columns = [c for c in df_subject_features.columns if c not in ["subject", "cohort"]]
X = df_subject_features[feature_columns].values
y = df_subject_features["cohort"].values


# --- Random Forest ---
rf = RandomForestClassifier(
    n_estimators=500,
    random_state=42,
    class_weight="balanced"
)

# Cross-validation: leave-one-subject-out (since only ~36 subjects)
loo = LeaveOneOut()
scores = []
for train_idx, test_idx in loo.split(X):
    rf.fit(X[train_idx], y[train_idx])
    scores.append(rf.score(X[test_idx], y[test_idx]))

#print("Leave-One-Out CV Accuracy:", np.mean(scores))

# --- Fit on full dataset for feature importance ---
rf.fit(X, y)

# Permutation importance
perm_importance = permutation_importance(rf, X, y, n_repeats=30, random_state=42)

importance = pd.DataFrame({
    "feature": feature_columns,
    "importance_mean": perm_importance.importances_mean,
    "importance_std": perm_importance.importances_std
}).sort_values("importance_mean", ascending=False)

#print("\nPermutation importance:")
#print(importance)

def run_subjectwise_lasso_logreg(
    df,
    feature_cols=('ABeta_local','Tau_local','I_local','X_local',
                  'ABeta_global_dif','Tau_global_dif','I_global_dif','X_global_dif'),
    label_col='cohort',
    group_col='subject',
    n_splits=5,
    Cs=10,   # number of C values to test on log scale
    max_iter=5000
):
    """
    Subject-wise LASSO logistic regression framework.
    """
    # restrict to available features
    feature_cols = [c for c in feature_cols if c in df.columns]
    X_all = df[feature_cols].to_numpy()
    y_all = df[label_col].to_numpy()
    groups_all = df[group_col].to_numpy()

    # subject-wise CV
    cv = GroupKFold(n_splits=n_splits)

    # pipeline with scaling + logistic regression
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('logreg', LogisticRegressionCV(
            Cs=np.logspace(-3, 3, Cs),
            cv=cv.split(X_all, y_all, groups_all),
            penalty='l1',
            solver='saga',
            class_weight='balanced',
            max_iter=max_iter,
            scoring='accuracy',
            n_jobs=-1
        ))
    ])

    # fit model
    pipe.fit(X_all, y_all)

    # parcel-level predictions (subject-wise CV already enforced in inner loop)
    y_pred = pipe.predict(X_all)

    parcel_acc = accuracy_score(y_all, y_pred)

    # subject-level majority vote
    subj_preds = defaultdict(list)
    subj_trues = {}
    for yi, yp, sid in zip(y_all, y_pred, groups_all):
        subj_preds[sid].append(yp)
        subj_trues[sid] = yi

    y_true_subj, y_pred_subj = [], []
    for sid, preds in subj_preds.items():
        maj = Counter(preds).most_common(1)[0][0]
        y_pred_subj.append(maj)
        y_true_subj.append(subj_trues[sid])

    subject_acc = accuracy_score(y_true_subj, y_pred_subj)

    # Diagnostics
    print("Parcel-level accuracy:", parcel_acc)
    print("Subject-level accuracy:", subject_acc)
    print("Classification report:")
    print(classification_report(y_all, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_all, y_pred, labels=np.unique(y_all)))
    print("Chosen C:", pipe.named_steps['logreg'].C_)

    return pipe, y_pred, y_pred_subj

# results = run_subjectwise_svm(
#     df_cohort)


import statsmodels.formula.api as smf

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS


def hierarchical_model(ABeta_local, Tau_local, ABeta_global, Tau_global, subject_idx, FDT_I=None):
    # Hyperpriors for random intercepts
    mu_a = numpyro.sample("mu_a", dist.Normal(0, 1))
    sigma_a = numpyro.sample("sigma_a", dist.Exponential(1.0))
    
    # Random intercept per subject
    a_subject = numpyro.sample("a_subject", dist.Normal(mu_a, sigma_a).expand([n_subjects]))
    
    # Fixed effects
    beta_ABeta = numpyro.sample("beta_ABeta", dist.Normal(0, 1))
    beta_Tau = numpyro.sample("beta_Tau", dist.Normal(0, 1))
    beta_inter = numpyro.sample("beta_inter", dist.Normal(0, 1))
    beta_ABeta_global = numpyro.sample("beta_ABeta_global", dist.Normal(0, 1))
    beta_Tau_global = numpyro.sample("beta_Tau_global", dist.Normal(0, 1))
    
    sigma = numpyro.sample("sigma", dist.Exponential(1.0))
    
    mu = (
        a_subject[subject_idx]
        + beta_ABeta * ABeta_local
        + beta_Tau * Tau_local
        + beta_inter * ABeta_local * Tau_local
        + beta_ABeta_global * ABeta_global
        + beta_Tau_global * Tau_global
    )
    
    numpyro.sample("obs", dist.Normal(mu, sigma), obs=FDT_I)

# Run MCMC
# nuts_kernel = NUTS(hierarchical_model)
# mcmc = MCMC(nuts_kernel, num_warmup=1000, num_samples=2000, num_chains=2)
# mcmc.run(jax.random.PRNGKey(0), ABeta_local, Tau_local, ABeta_global, Tau_global, subject_idx_array, FDT_I)
# mcmc.print_summary()

import numpy as np
import pandas as pd
from sklearn.decomposition import SparsePCA
from scipy.stats import ttest_ind

# --- 1. Reshape data: subject × (parcel × feature) ---
df_wide = (
    df_cohort.pivot(index="subject", columns="parcel", values=["ABeta_local", "Tau_local", "I_local", "X_local"])
)
df_wide.columns = [f"{feat}_parcel{p}" for feat, p in df_wide.columns]
X = df_wide.values  # shape: n_subjects × (n_parcels * 4)

print("Data shape for SparsePCA:", X.shape)

# --- 2. Run Sparse PCA ---
n_components = 5   # number of subnetworks
alpha = 0.01          # sparsity (increase for fewer parcels per component)

spca = SparsePCA(n_components=n_components, alpha=alpha, random_state=42)
X_spca = spca.fit_transform(X)     # subject × component scores
components = spca.components_      # component × (parcel × feature) loadings

print("Component scores shape:", X_spca.shape)
print("Component loadings shape:", components.shape)

# --- 3. Inspect top contributing parcels/features ---
for comp_idx in range(n_components):
    loading_vec = components[comp_idx]
    top_idx = np.argsort(np.abs(loading_vec))[::-1][:10]  # top 10 features
    top_features = [df_wide.columns[i] for i in top_idx]
    print(f"\nTop features for Component {comp_idx+1}:")
    print(top_features)

# --- 4. Group-level statistical testing ---
df_scores = pd.DataFrame(X_spca, index=df_wide.index, columns=[f"Comp{i+1}" for i in range(n_components)])
df_scores["group"] = df_subject_features.set_index("subject").loc[df_wide.index, "cohort"]

for c in df_scores.columns[:-1]:
    g0 = df_scores[df_scores["group"] == 0][c]
    g1 = df_scores[df_scores["group"] == 2][c]
    stat, p = ttest_ind(g0, g1)
    print(f"{c}: t={stat:.2f}, p={p:.4f}")

