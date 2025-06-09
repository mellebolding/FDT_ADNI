import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def compute_FC_1sub(timeseries_data, trim=10):
    """
    Compute empirical functional connectivity (FC) matrices.

    Parameters:
        timeseries_data (ndarray): Shape (NSUB, NPARCELLS, NTIMEPOINTS) fMRI time series.
        trim (int): Number of time points to remove at the start
            and end of the time series.
    Returns:
        FC_empirical (list of ndarray): List of FC matrices for each subject.
    """
    NPARCELLS, NTIMEPOINTS = timeseries_data.shape
    indexN = np.arange(NPARCELLS)  # Select all brain regions

    ts = timeseries_data  # Extract subject data
    ts2 = ts[indexN, trim:-trim]  # Select regions and remove 10 time points from start and end
    FC_empirical = np.corrcoef(ts2)  # Compute Pearson correlation matrix

    return FC_empirical  # Returns the FC matrix

####################################################################

def compute_FC_1sub_vectorized(timeseries_data, trim=10):
    """
    Compute empirical functional connectivity (FC) matrix for a single subject.

    Parameters:
        timeseries_data (ndarray): Shape (NPARCELS, NTIMEPOINTS) fMRI time series.
        trim (int): Number of time points to remove at the start and end.
        
    Returns:
        FC_empirical (ndarray): Functional connectivity matrix of shape (NPARCELS, NPARCELS).
    """
    # Trim the time series (avoid unnecessary index selection)
    ts_trimmed = timeseries_data[:, trim:-trim]  # Shape: (NPARCELS, Tm)

    # Z-score normalization (element-wise)
    ts_mean = np.mean(ts_trimmed, axis=1, keepdims=True)
    ts_std = np.std(ts_trimmed, axis=1, keepdims=True)
    ts_zscored = (ts_trimmed - ts_mean) / (ts_std + 1e-8)  # Prevent division by zero

    # Compute correlation matrix using matrix multiplication (fast alternative to np.corrcoef)
    FC_empirical = np.einsum('ij,kj->ik', ts_zscored, ts_zscored) / (ts_trimmed.shape[1] - 1)

    return FC_empirical

####################################################################

def compute_FC_Nsub(timeseries_data, trim=10):
    """
    Computes the empirical functional connectivity (FC) matrix for each subject.
    
    Parameters:
        tsdata (ndarray): fMRI time series of shape (NSUB, NPARCELS, NTIMEPOINTS).
        trim (int): Number of time points to remove at the start and end of the time series.
        
    Returns:
        FC_empirical (ndarray): FC matrices of shape (NSUB, NPARCELS, NPARCELS).
    """
    NSUB, NPARCELS, NTIMEPOINTS = timeseries_data.shape
    FC_empirical = np.zeros((NSUB, NPARCELS, NPARCELS))  # FC for each subject
    indexN = np.arange(NPARCELS)

    for sub in range(NSUB):
        ts = timeseries_data[sub]  # Shape: (NPARCELS, NTIMEPOINTS)
        ts_selected = ts[indexN, trim:-trim]  # Trim time points
        FC_empirical[sub] = np.corrcoef(ts_selected)  # Compute correlation

    return FC_empirical

####################################################################

def estimate_FC_residual_1sub(fmri_timeseries, simulated_timeseries):
    """
    Estimate sigma using functional connectivity residuals.
    
    Parameters:
        fmri_timeseries (ndarray): Empirical fMRI time series of shape (N_regions, T).
        simulated_timeseries (ndarray): Simulated time series from the Hopf model.
    
    Returns:
        sigma_estimates (ndarray): Estimated sigma values for each brain region.
    """
    FC_empirical = compute_FC_1sub(fmri_timeseries)
    FC_simulated = compute_FC_1sub(simulated_timeseries)
    
    residuals = FC_empirical - FC_simulated
    residual_variance = np.var(residuals, axis=1)
    
    return np.sqrt(residual_variance)

####################################################################

def plot_FC_matrix(FC_matrix, title="Functional Connectivity Matrix", cmap="turbo", size=1, save_path=None, dpi=300):
    """
    Plots a functional connectivity matrix as a heatmap with simplified axis ticks.
    
    Parameters:
        FC_matrix (ndarray): N x N functional connectivity matrix.
        title (str): Title of the plot.
        cmap (str): Colormap for the heatmap.
        size (float): Scaling factor for figure size.
        save_path (str or None): If provided, the plot will be saved to this file path.
    """
    plt.figure(figsize=(4 * size, 3.5 * size))
    im = plt.imshow(FC_matrix, cmap=cmap)
    plt.colorbar(label="Connectivity Strength")
    plt.title(title)
    plt.xlabel("Region Index")
    plt.ylabel("Region Index")

    # Set ticks and labels
    n = FC_matrix.shape[0]
    tick_positions = np.arange(n)
    tick_labels = [str(tick_positions[0] + 1)] + [''] * (n - 2) + [str(tick_positions[-1] + 1)]
    plt.xticks(tick_positions, tick_labels)
    plt.yticks(tick_positions, tick_labels)

    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    else: plt.show()
    plt.close()


def plot_FC_matrices(FC1, FC2, title1="FC Matrix 1", title2="FC Matrix 2", cmap="turbo", size=1, save_path=None, dpi=300):
    """
    Plots two functional connectivity matrices side by side, each with its own colorbar.
    
    Parameters:
        FC1 (ndarray): First N x N functional connectivity matrix.
        FC2 (ndarray): Second N x N functional connectivity matrix.
        title1 (str): Title for the first matrix.
        title2 (str): Title for the second matrix.
        cmap (str): Colormap for the heatmaps.
        size (float): Scaling factor for figure size.
    """
    fig, axes = plt.subplots(1, 2, figsize=(8 * size, 3.5 * size))

    # Helper function to set ticks and labels for a given axis
    def set_ticks_and_labels(ax, matrix):
        n = matrix.shape[0]  # Number of regions
        tick_positions = np.arange(n)  # Positions for ticks (0 to n-1)
        # tick_labels = np.arange(1, n + 1)  # Labels starting from 1 to n
        tick_labels = [str(tick_positions[0]+1)] + [''] * (len(tick_positions) - 2) + [str(tick_positions[-1]+1)]

        # Set ticks at the center of each square
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)
        
        # Set tick labels
        ax.set_xticklabels(tick_labels)
        ax.set_yticklabels(tick_labels)

    # Plot the first matrix
    im1 = axes[0].imshow(FC1, cmap=cmap)  # , vmin=-1, vmax=1)
    axes[0].set_title(title1)
    axes[0].set_xlabel("Region Index")
    axes[0].set_ylabel("Region Index")
    cbar1 = fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    cbar1.set_label("Connectivity Strength")
    set_ticks_and_labels(axes[0], FC1)

    # Plot the second matrix
    im2 = axes[1].imshow(FC2, cmap=cmap)  # , vmin=-1, vmax=1)
    axes[1].set_title(title2)
    axes[1].set_xlabel("Region Index")
    cbar2 = fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    cbar2.set_label("Connectivity Strength")
    set_ticks_and_labels(axes[1], FC2)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')

    else: plt.show()
    plt.close()

def plot_avg_X_vs_N(RESULTS):
    """
    Plots the average of X over the first N simulations as a function of N for each parcel.

    Parameters:
        RESULTS (ndarray): Array of shape (NSIM, NPARCELS) storing the quantity X.
    """
    NSIM, NPARCELS = RESULTS.shape
    N_values = np.arange(1, NSIM + 1)  # Values of N (number of simulations)
    
    # Compute cumulative mean for each parcel as a function of N
    avg_X = np.cumsum(RESULTS, axis=0) / N_values[:, None]

    # Plot
    plt.figure(figsize=(8, 5))
    for p in range(NPARCELS):
        plt.plot(N_values, avg_X[:, p]) #, label=f"Parcel {p+1}", alpha=0.6)

    plt.xlabel("Number of Simulations (N)")
    plt.ylabel("Average X over first N simulations")
    plt.title("Convergence of X as a function of N")
    # plt.legend(loc="upper right", fontsize=8, ncol=2, frameon=False)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()