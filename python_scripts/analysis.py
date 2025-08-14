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
from nilearn import surface, datasets, plotting
import nibabel as nib
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

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
        fig_name = f"box_parcel_Itmax_N{NPARCELLS}_{NOISE_TYPE}"
        save_path = os.path.join(FDT_parcel_subfolder, fig_name)
        p_values.plotComparisonAcrossLabels2(
            resI_Itmax,
            custom_test=statannotations_permutation.stat_permutation_test,
            columnLables=['HC', 'MCI', 'AD'],
            graphLabel=f'FDT I(tmax, 0) Parcels {NOISE_TYPE}',
            save_path=save_path
        )
        plt.rcParams.update({'font.size': 15})
        fig_name = f"box_parcel_norm1_N{NPARCELLS}_{NOISE_TYPE}"
        save_path = os.path.join(Inorm1_group_subfolder, fig_name)
        p_values.plotComparisonAcrossLabels2(
            resI_norm1,
            custom_test=statannotations_permutation.stat_permutation_test,
            columnLables=['HC', 'MCI', 'AD'],
            graphLabel=f'FDT I Norm1 Parcels {NOISE_TYPE}',
            save_path=save_path
        )
        plt.rcParams.update({'font.size': 15})
        fig_name = f"box_parcel_norm2_N{NPARCELLS}_{NOISE_TYPE}"
        save_path = os.path.join(Inorm2_group_subfolder, fig_name)
        p_values.plotComparisonAcrossLabels2(
            resI_norm2,
            custom_test=statannotations_permutation.stat_permutation_test,
            columnLables=['HC', 'MCI', 'AD'],
            graphLabel=f'FDT I Norm2 Parcels {NOISE_TYPE}',
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
        fig_name = f"violin_subject_N{NPARCELLS}_{NOISE_TYPE}"
        save_path = os.path.join(FDT_subject_subfolder, fig_name)
        plot_violins_HC_MCI_AD(
            ax=ax,
            data=data_subjects_Itmax,
            font_scale=1.4,
            metric='I(t=tmax,s=0) [Subject mean]',
            point_size=5,
            xgrid=False,
            plot_title=f'FDT I(tmax, 0) — Mean per subject per group {NOISE_TYPE}',
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
            plot_title=f'FDT I Norm1 — Mean per subject per group {NOISE_TYPE}',
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
            plot_title=f'FDT I Norm2 — Mean per subject per group {NOISE_TYPE}',
            saveplot=1,
            filename=save_path,
            dpi=300
        )

def figures_barplot_parcels(option,I_tmax_group,NPARCELLS):
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
    fig_name = f"barplot_parcel_{option}_N{NPARCELLS}_{NOISE_TYPE}"
    save_path = os.path.join(FDT_parcel_subfolder, fig_name)
    bottom = np.zeros(NPARCELLS)  # start at zero for stacking
    for i in range(3):
        plt.bar(range(NPARCELLS), I_group[i], color=colors[i], label=f'{["HC", "MCI", "AD"][i]}', alpha=0.45)
        #bottom += I_group[i]

    plt.xlabel('Parcel')
    plt.ylabel(f'{option}')
    plt.title(f'{option} for Parcels {NOISE_TYPE}')
    plt.legend()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_means_per_RSN(name, I_tmax_group, NPARCELLS):

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
    ax.set_title(f'Mean {name} per RSN (first {NPARCELLS} parcels) {NOISE_TYPE}')
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
    ax.set_title(f'Mean {nameI} for {nameRSN} RSN {NOISE_TYPE}')

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

def left_right_brain_map(name,I_tmax_group,COND,NPARCELLS):
    """
    Visualizes the group differences in I(tmax, 0) on a brain map.
    """
    nii_path = os.path.join('ADNI-A_DATA', 'MNI_Glasser_HCP_v1.0.nii.gz')
    parcel_img = nib.load(nii_path)
    parcel_data = parcel_img.get_fdata()
    group_map = np.zeros_like(parcel_data)

    group_values = I_tmax_group[COND,:]

    for i in range(NPARCELLS):
        group_map[parcel_data == i + 1] = group_values[i]

    group_img = nib.Nifti1Image(group_map, affine=parcel_img.affine)
    fsaverage = datasets.fetch_surf_fsaverage()

    texture_left = surface.vol_to_surf(group_img, fsaverage.pial_left)
    texture_right = surface.vol_to_surf(group_img, fsaverage.pial_right)

    vmin = np.min(group_values)
    vmax = np.max(group_values)

    fig = plt.figure(figsize=(10, 5))
    fig_name = f"left_right_brain{name}_N{NPARCELLS}_{NOISE_TYPE}"
    save_path = os.path.join(FDT_parcel_subfolder, fig_name)

    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    plotting.plot_surf_stat_map(fsaverage.pial_left, texture_left,
                                hemi='left', title = f'{name} Left',
                                view='lateral',
                                colorbar=False, cmap='viridis',
                                bg_map=fsaverage.sulc_left,
                                vmin=vmin, vmax=vmax,
                                axes=ax1, darkness=None)

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    plotting.plot_surf_stat_map(fsaverage.pial_right, texture_right,
                                hemi='right', title = f'{name} Right',
                                view='lateral',
                                colorbar=False, cmap='viridis',
                                bg_map=fsaverage.sulc_right,
                                vmin=vmin, vmax=vmax,
                                axes=ax2, darkness=None)

    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])

    # Define position: [left, bottom, width, height] in figure coordinates (0 to 1)
    cbar_ax = fig.add_axes([0.47, 0.25, 0.02, 0.5])  # adjust as needed
    # Create the colorbar manually in that position
    cbar = plt.colorbar(sm, cax=cbar_ax)
    # cbar.set_label("Group Difference")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def brain_map_3D(name, I_tmax_group, COND, NPARCELLS):

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
    view = plotting.view_surf(surf_mesh=mesh_both,
                            surf_map=surf_map,
                            cmap='viridis',
                            vmin=np.min(surf_map),     # Minimum value of colorbar
                            vmax=np.max(surf_map),     # Maximum value of colorbar
                            symmetric_cmap=False,
                            colorbar=True,
                            darkness=None,
                            title=f'{name}')
    view.open_in_browser()  # or just `view` if using Jupyter
    view.save_as_html(f'surface_plot_{name}_N{NPARCELLS}_{NOISE_TYPE}.html')
    view

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
NOISE_TYPE = "HOMO"

all_values = load_appended_records(
    filepath=os.path.join(FDT_values_subfolder, f"FDT_values_{NPARCELLS}_{NOISE_TYPE}.npz")
)
I_tmax_group = np.squeeze(np.array(get_field(all_values, "I_tmax", filters={"level": "group"})),axis=0)
I_norm1_group = np.squeeze(np.array(get_field(all_values, "I_norm1", filters={"level": "group"})), axis=0)
I_norm2_group = np.squeeze(np.array(get_field(all_values, "I_norm2", filters={"level": "group"})), axis=0)
I_tmax_sub = np.squeeze(np.array(get_field(all_values, "I_tmax", filters={"level": "subject"})), axis=0)
I_norm1_sub = np.squeeze(np.array(get_field(all_values, "I_norm1", filters={"level": "subject"})), axis=0)
I_norm2_sub = np.squeeze(np.array(get_field(all_values, "I_norm2", filters={"level": "subject"})), axis=0)


# figures_I_tmax_norm1_norm2(group=True, subject=False, I_tmax=I_tmax_group, I_norm1=I_norm1_group, I_norm2=I_norm2_group)
# figures_I_tmax_norm1_norm2(group=False, subject=True, I_tmax=I_tmax_sub, I_norm1=I_norm1_sub, I_norm2=I_norm2_sub)

# figures_barplot_parcels('I_tmax',I_tmax_group, NPARCELLS)
# figures_barplot_parcels('I_norm1', I_norm1_group, NPARCELLS)
# figures_barplot_parcels('I_norm2', I_norm2_group, NPARCELLS)


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



# plot_means_per_RSN('I_tmax', I_tmax_group, NPARCELLS)
# plot_means_per_RSN('I_norm1', I_norm1_group, NPARCELLS)
# plot_means_per_RSN('I_norm2', I_norm2_group, NPARCELLS)

# plot_means_per_subjects_per_RSN(SomMot, I_tmax_sub, 'SomMot', 'I_tmax', NPARCELLS)
# plot_means_per_subjects_per_RSN(Vis, I_tmax_sub, 'Vis', 'I_tmax', NPARCELLS)
#plot_means_per_subjects_per_RSN(Limbic, I_tmax_sub, 'Limbic', 'I_tmax', NPARCELLS)
# #...

###### VISUALIZATION ######
# left_right_brain_map('I_tmax_HC', I_tmax_group, 0, NPARCELLS)
# left_right_brain_map('I_tmax_MCI', I_tmax_group, 1, NPARCELLS)
# left_right_brain_map('I_tmax_AD', I_tmax_group, 2, NPARCELLS)

# brain_map_3D(f'I_tmax_HC_{NOISE_TYPE}', I_tmax_group, 0, NPARCELLS)
# brain_map_3D(f'I_tmax_MCI_{NOISE_TYPE}', I_tmax_group, 1, NPARCELLS)
# brain_map_3D(f'I_tmax_AD_{NOISE_TYPE}', I_tmax_group, 2, NPARCELLS)
brain_map_3D(f'I_tmax_HC_{NOISE_TYPE}_0', I_tmax_sub[0], 0, NPARCELLS)
brain_map_3D(f'I_tmax_HC_{NOISE_TYPE}_1', I_tmax_sub[0], 1, NPARCELLS)
brain_map_3D(f'I_tmax_HC_{NOISE_TYPE}_2', I_tmax_sub[0], 2, NPARCELLS)
brain_map_3D(f'I_tmax_HC_{NOISE_TYPE}_3', I_tmax_sub[0], 3, NPARCELLS)
brain_map_3D(f'I_tmax_HC_{NOISE_TYPE}_4', I_tmax_sub[0], 4, NPARCELLS)
brain_map_3D(f'I_tmax_HC_{NOISE_TYPE}_5', I_tmax_sub[0], 5, NPARCELLS)

# I_group is shape (3, 379)
# example: I_group[0] = HC values, I_group[1] = MCI values, I_group[2] = AD values

# diffs = np.max(I_tmax_group, axis=0) - np.min(I_tmax_group, axis=0)  # range per parcel
# top_n = 10  # how many top parcels you want
# top_parcels = np.argsort(diffs)[::-1][:top_n]  # indices of largest differences

# print("Top parcels with largest group differences:")
# for idx in top_parcels:
#     print(f"Parcel {idx}: range = {diffs[idx]:.4f}, values = {I_tmax_group[:, idx]}")

import numpy as np
import matplotlib.pyplot as plt


groups = ["HC", "MCI", "AD"]
colors = ["tab:blue", "tab:orange", "tab:green"]

# 1. Compute range per parcel
diffs = np.max(I_norm2_group, axis=0) - np.min(I_norm2_group, axis=0)

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
        I_norm2_group[i, top_parcels],
        width=bar_width,
        label=group,
        color=colors[i]
    )

# 4. Set labels
print([Parcel_names.get(idx+1, f"Parcel {idx+1}") for idx in top_parcels])
ax.set_xticks(x + bar_width)
ax.set_xticklabels([Parcel_names.get(idx+1, f"Parcel {idx+1}") for idx in top_parcels], rotation=45, ha="right")
ax.set_ylabel("Value")
ax.set_title("Top parcels with largest between-group differences")
ax.legend()

plt.tight_layout()
plt.show()

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
            I_norm2_sub[i, subj, top6_parcels],
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
plt.show()

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
# from sklearn.model_selection import cross_val_score, StratifiedKFold
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
# from scipy.stats import f_oneway

# # Assume you have already computed the top 18 parcels:
# # e.g., top_parcel_indices = [list of 18 indices]

# # Select only those parcels
# X_top18 = I_tmax_sub[:, :, top_parcels]  # shape: (3, n_subs, 18)

# # Reshape to (n_subjects, n_parcels)
# n_groups, n_subs, n_parcels = X_top18.shape

# X = X_top18.reshape(-1, n_parcels)  # (total_subjects, 18)
# # Remove any subject (row) that has NaN in any of the top 18 parcels
# mask = ~np.isnan(X).any(axis=1)
# X = X[mask]


# # Create group labels
# groupss = np.array([[g]*n_subs for g in groups]).flatten()  # length = total_subjects
# groupss = groupss[mask]
# # -----------------
# # PCA for visualization
# # -----------------
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt

# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X)

# plt.figure(figsize=(8,6))
# colors = {'HC':'tab:blue', 'MCI':'tab:orange', 'AD':'tab:green'}
# for g in np.unique(groupss):
#     idx = groupss == g
#     plt.scatter(X_pca[idx,0], X_pca[idx,1], label=g, color=colors[g], alpha=0.7)
# plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
# plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
# plt.title("PCA on top 18 parcels by group difference")
# plt.legend()
# plt.tight_layout()
# plt.show()

# # -----------------
# # ANOVA per parcel
# # -----------------
# from scipy.stats import f_oneway
# parcel_names_top18 = [Parcel_names[i] for i in top_parcels]  # actual names

# anova_results = []
# for i in range(n_parcels):
#     vals_per_group = [X_top18[g, :, i] for g in range(n_groups)]
#     f_val, p_val = f_oneway(*vals_per_group)
#     anova_results.append((parcel_names_top18[i], f_val, p_val))

# anova_df = pd.DataFrame(anova_results, columns=["Parcel", "F-value", "p-value"])
# anova_df.sort_values("p-value", inplace=True)
# print("\nTop parcels by ANOVA p-value:")
# print(anova_df)

# # -----------------
# # LDA classification
# # -----------------
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
# from sklearn.model_selection import cross_val_score, StratifiedKFold

# clf = LDA()
# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# scores = cross_val_score(clf, X, groupss, cv=cv)

# print(f"\nLDA classification accuracy (5-fold CV): {np.mean(scores)*100:.2f}% ± {np.std(scores)*100:.2f}%")
