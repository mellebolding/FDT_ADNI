import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm, solve_sylvester
from scipy.signal import correlate
#from IPython.display import display, clear_output

def LinHopf_Ceff_sigma_a_fitting_numba(tsdata, C, NPARCELS, TR, f_diff, sigma, a=-0.02, Tau=1,
                                     fit_Ceff=True, competitive_coupling=False,
                                     fit_sigma=True, sigma_reset=False,
                                     fit_a=True
                                     epsFC_Ceff=8e-5, epsCOVtau_Ceff=3e-5, epsFC_sigma=8e-5, epsCOVtau_sigma=3e-5, 
                                     epsFC_a=8e-5, epsCOVtau_a=3e-5,
                                     MAXiter=10000, error_tol=5e-4, patience=2, learning_rate_factor=0.8,
                                     Ceff_norm=True, maxC=0.2,
                                     iter_check=50, plot_evol=False, plot_evol_last=False):
    """
    Fits the "averaged" G.Cij matrix AND the sigma values using the FC.
    Some parts are optimized with numba

    Parameters:
        tsdata: Empirical Time Series data (3D array: [NSUB, NPARCELS, timepoints] or 2D array: [NPARCELS, timepoints])
        C: Structural connectivity matrix
        NPARCELS: Number of parcels
        TR: Repetition time
        f_diff: Nodes frequencies
        sigma: Noise variance
        a: bifurcation parameter
        Tau: Time lag
        fit_Ceff: Flag for fitting Ceff
        competitive_coupling: Flag for competitive coupling
        fit_sigma: Flag for fitting sigma
        sigma_reset: Flag for resetting sigma to sigma_ini if sigma_new < 0
        epsFC_Ceff: Learning rate
        epsCOVtau_Ceff: Learning rate
        epsFC_sigma: Learning rate
        epsCOVtau_sigma: Learning rate
        MAXiter: Maximum number of iterations
        error_tol: Tolerance for convergence
        patience: Number cycles of 100 iterations below error_tol before stopping
        learning_rate_factor: Factor that multiplies the learning rate inside the patience loop
        Ceff_norm: Flag for normalizing the Ceff
        maxC: Normalization factor for Ceff
        iter_check: Number of iterations to check error
        plot_evol: Flag for plotting evolution of error

    Returns:
        Ceff_fit: Effective connectivity matrix
        sigma_fit: Standard deviation of noise
        FCemp: Empirical functional connectivity matrix
        FCsim: Simulated functional connectivity matrix
        error_iter: List of errors at every 100 iterations
    """

    indexN = np.arange(NPARCELS)  # Cortical areas
    N = len(indexN)

    FCPB = []
    COVPB = []
    COVtauPB = []

    NSUB = tsdata.shape[0]
    if tsdata.ndim == 2:
        tsdata = np.expand_dims(tsdata, axis=0)
        NSUB = 1
        
    for sub in range(NSUB):
        ts = tsdata[sub, :, :].copy()    # Extract subject's time series
        ts2 = ts[indexN, 10:-10].copy()  # Remove edges

        ## Empirical FC(0)
        FCemp = np.corrcoef(ts2)
        FCPB.append(FCemp)
        COVemp = np.cov(ts2)
        COVPB.append(COVemp)

        ## Empirical COV(tau)
        tst = ts2.T
        COVtauemp = np.zeros((N, N))
        sigratio = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                sigratio[i, j] = 1 / np.sqrt(COVemp[i, i]) / np.sqrt(COVemp[j, j])
                clag, lags = xcov(tst[:, i], tst[:, j], Tau, norm='none') #OK
                indx = np.where(lags == Tau)[0][0]
                COVtauemp[i, j] = clag[indx] / tst.shape[0]
        COVtauemp *= sigratio
        COVtauPB.append(COVtauemp)

    ## Group FC, COV, and COVtau
    FCemp = np.mean(FCPB, axis=0)
    COVemp = np.mean(COVPB, axis=0)
    COVtauemp = np.mean(COVtauPB, axis=0)
    COVemp_diag = np.diag(COVemp)

    ### Optimization loop ###
    Ceff_new = C.copy()
    Ceff_previous = Ceff_new.copy()
    sigma_ini = sigma * np.ones(NPARCELS)
    sigma_new = sigma_ini.copy()
    sigma_previous = sigma_new.copy()
    a_ini = a * np.ones(NPARCELS)
    a_new = a_ini.copy()
    a_previous = a_new.copy()

    error_old = 1e5
    error_tol_old = 1e5
    errorFC_iter = []
    errorCOVtau_iter = []
    error_iter = []
    patience_sum = 0

    for iter in range(1, MAXiter + 1):
        ### Linear Hopf FC
        FCsim, COVsim, COVsimtotal, A = hopf_int(Ceff_new, f_diff, sigma_new, a_new)

        # COVtausim = (expm((Tau * TR) * A) @ COVsimtotal)[:N, :N].copy()
        COVtausim = (exp_scaling_squaring((Tau * TR) * A) @ COVsimtotal)[:N, :N].copy()
        
        sigratiosim = compute_sigratio_from_cov(COVsim)
        COVtausim *= sigratiosim

        if iter % iter_check == 0:
            errorFC_now = np.mean((FCemp - FCsim) ** 2)
            errorFC_iter.append(errorFC_now)
            errorCOVtau_now = np.mean((COVtauemp - COVtausim) ** 2)
            errorCOVtau_iter.append(errorCOVtau_now)

            error_now = errorFC_now + errorCOVtau_now
            error_iter.append(error_now)

            if  error_old < error_now:
                #print(f"Iter {iter:4d}/{MAXiter} ; error: {error_now:.7f} ; error_tol: {error_tol_now:.7f} | error_old < error_now --> EXIT (return previous sigma)")
                Ceff_fit = Ceff_previous
                sigma_fit = sigma_previous
                a_fit = a_previous
                break
            error_tol_now = (error_old - error_now) / error_now
            if error_tol_now < error_tol or np.abs(error_tol_old - error_tol_now) < error_tol/10:
                if patience_sum >= patience:
                    #print(f"Iter {iter:4d}/{MAXiter} ; error: {error_now:.7f} ; error_tol: {error_tol_now:.7f} ; patience_sum: {patience_sum} | Achieved convergence --> EXIT (return present sigma)")
                    Ceff_fit = Ceff_new
                    sigma_fit = sigma_new
                    break
                patience_sum += 1
                ## Modify learning rate
                epsFC_Ceff *= learning_rate_factor
                epsCOVtau_Ceff *= learning_rate_factor
                epsFC_sigma *= learning_rate_factor
                epsCOVtau_sigma *= learning_rate_factor
                epsFC_a *= learning_rate_factor
                epsCOVtau_a *= learning_rate_factor
            else:
                patience_sum = 0
            #print(f"Iter {iter:4d}/{MAXiter} ; error: {error_now:.7f} ; error_tol: {error_tol_now:.7f} ; patience_sum: {patience_sum}")
            error_old = error_now
            error_tol_old = error_tol_now

            if plot_evol==True:
                clear_output(wait=True)
                plt.figure(figsize=(18, 5))
                plt.subplot(141)
                plt.title('error_iter')
                plt.plot(np.arange(1,len(error_iter)+1)*iter_check, error_iter,'o-')
                plt.plot(np.arange(1,len(error_iter)+1)*iter_check, errorFC_iter,'.-', label=f'errorFC_iter')
                plt.plot(np.arange(1,len(error_iter)+1)*iter_check, errorCOVtau_iter,'.-', label=f'errorCOVtau_iter')
                plt.xlabel('Iteration')
                plt.legend()
                plt.subplot(142)
                plt.title('errorFC_iter')
                plt.plot(np.arange(1,len(errorFC_iter)+1)*iter_check, errorFC_iter,'o-', color='tab:orange', 
                         label=f'epsFC_Ceff={epsFC_Ceff:.3e} \nepsFC_sigma={epsFC_sigma:.3e}')
                plt.xlabel('Iteration')
                plt.legend()
                plt.subplot(143)
                plt.title('errorCOVtau_iter')
                plt.plot(np.arange(1,len(errorCOVtau_iter)+1)*iter_check, errorCOVtau_iter,'o-', color='tab:green', 
                         label=f'epsCOVtau_Ceff={epsCOVtau_Ceff:.3e} \nepsCOVtau_sigma={epsCOVtau_sigma:.3e}')
                plt.xlabel('Iteration')
                plt.legend()
                plt.subplot(144)
                plt.title('sigma')
                plt.plot(np.arange(1,N+1),sigma_previous,'o-',label='previous')
                plt.plot(np.arange(1,N+1),sigma_new,'.-', label='new')
                plt.xlabel('Parcel')
                plt.legend()
                plt.show()

        ### Learning rule Ceff
        if fit_Ceff:
            Ceff_previous = Ceff_new.copy()
            Ceff_new = update_Ceff(Ceff_previous, C, FCemp, FCsim, COVtauemp, COVtausim,
                                   epsFC_Ceff, epsCOVtau_Ceff, maxC, Ceff_norm, competitive_coupling)

        ### Learning rule sigma
        if fit_sigma:
            sigma_previous = sigma_new.copy()
            sigma_new = update_sigma(sigma_previous, sigma_ini,
                                     FCemp, FCsim, COVtauemp, COVtausim,
                                     epsFC_sigma, epsCOVtau_sigma, sigma_reset)
            ### sigma re-normalization
            _, COVsim, _, _ = hopf_int(Ceff_new, f_diff, sigma_new, a)
            COVsim_diag = np.diag(COVsim)
            normalization_factor = np.sum(COVemp_diag) / np.sum(COVsim_diag)
            sigma_new *= np.sqrt(normalization_factor)
        
        if fit_a:
            a_previous = a_new.copy()
            a_new = update_a(a_previous, a_ini,FCemp, FCsim, COVtauemp, COVtausim,epsFC_a, epsCOVtau_a)


    if iter == MAXiter:
        #print('Reached max. iterations:',MAXiter)
        Ceff_fit = Ceff_new
        sigma_fit = sigma_new
        a_fit = a_new
    

    FCsim, _, _, _ = hopf_int(Ceff_fit, f_diff, sigma_fit, a_fit)

    if plot_evol_last==True:
        plt.figure(figsize=(18, 5))
        plt.subplot(141)
        plt.title('error_iter')
        plt.plot(np.arange(1,len(error_iter)+1)*iter_check, error_iter,'o-')
        plt.plot(np.arange(1,len(error_iter)+1)*iter_check, errorFC_iter,'.-', label=f'errorFC_iter')
        plt.plot(np.arange(1,len(error_iter)+1)*iter_check, errorCOVtau_iter,'.-', label=f'errorCOVtau_iter')
        plt.xlabel('Iteration')
        plt.legend()
        plt.subplot(142)
        plt.title('errorFC_iter')
        plt.plot(np.arange(1,len(errorFC_iter)+1)*iter_check, errorFC_iter,'o-', color='tab:orange', 
                    label=f'epsFC_Ceff={epsFC_Ceff:.3e} \nepsFC_sigma={epsFC_sigma:.3e}')
        plt.xlabel('Iteration')
        plt.legend()
        plt.subplot(143)
        plt.title('errorCOVtau_iter')
        plt.plot(np.arange(1,len(errorCOVtau_iter)+1)*iter_check, errorCOVtau_iter,'o-', color='tab:green', 
                    label=f'epsCOVtau_Ceff={epsCOVtau_Ceff:.3e} \nepsCOVtau_sigma={epsCOVtau_sigma:.3e}')
        plt.xlabel('Iteration')
        plt.legend()
        plt.subplot(144)
        plt.title('sigma')
        plt.plot(np.arange(1,N+1),sigma_previous,'o-',label='previous')
        plt.plot(np.arange(1,N+1),sigma_new,'.-', label='new')
        plt.xlabel('Parcel')
        plt.legend()
        plt.show()

    return Ceff_fit, sigma_fit, a_fit, FCemp, FCsim, error_iter, errorFC_iter, errorCOVtau_iter


from numba import njit, prange
@njit
def compute_sigratio_from_cov(COV):
    N = COV.shape[0]
    sigratio = np.zeros((N, N))
    for i in range(N):
        sqrt_ii = np.sqrt(COV[i, i])
        for j in range(N):
            sigratio[i, j] = 1.0 / (sqrt_ii * np.sqrt(COV[j, j]))
    return sigratio

@njit(parallel=True)
def update_Ceff(Ceff_previous, C, FCemp, FCsim, COVtauemp, COVtausim,
                epsFC_Ceff, epsCOVtau_Ceff, maxC, Ceff_norm, competitive_coupling=False):
    N = Ceff_previous.shape[0]
    Ceff_new = np.zeros_like(Ceff_previous)

    for i in prange(N):
        for j in range(N):
            if C[i, j] > 0 or j == N - i - 1:
                Ceff_new[i, j] = Ceff_previous[i, j] + \
                                 epsFC_Ceff * (FCemp[i, j] - FCsim[i, j]) + \
                                 epsCOVtau_Ceff * (COVtauemp[i, j] - COVtausim[i, j])

                if Ceff_new[i, j] < 0 and competitive_coupling == False:
                    Ceff_new[i, j] = 0
                # Safeguard for nan (when competitive_coupling == True)
                if not np.isfinite(Ceff_new[i, j]):
                    Ceff_new[i, j] = 0.0

    if Ceff_norm:
        max_val = np.max(np.abs(Ceff_new))
        if max_val > 0:
            Ceff_new *= maxC / max_val

    return Ceff_new

@njit(parallel=True)
def update_sigma(sigma_previous, sigma_ini,
                 FCemp, FCsim, COVtauemp, COVtausim,
                 epsFC_sigma, epsCOVtau_sigma, sigma_reset=False):
    N = sigma_previous.shape[0]
    sigma_new = np.zeros(N)

    for i in prange(N):
        grad_FC = np.sum(FCemp[i, :] - FCsim[i, :])
        grad_COVtau = np.sum(COVtauemp[i, :] - COVtausim[i, :])
        
        sigma_new[i] = sigma_previous[i] - epsFC_sigma * grad_FC - epsCOVtau_sigma * grad_COVtau
        
        if sigma_new[i] < 0 and sigma_reset==True:
            sigma_new[i] = sigma_ini[i]
        elif sigma_new[i] < 0 and sigma_reset==False:
            sigma_new[i] = 0

    return sigma_new

def update_a(a_previous, a_ini, FCemp, FCsim, COVtauemp, COVtausim,
                epsFC_a, epsCOVtau_a):
    N = len(a_previous)
    a_new = np.zeros(N)

    for i in range(N):
        grad_FC = np.sum(FCemp[i, :] - FCsim[i, :])
        grad_COVtau = np.sum(COVtauemp[i, :] - COVtausim[i, :])
        
        a_new[i] = a_previous[i] - epsFC_a * grad_FC - epsCOVtau_a * grad_COVtau
        
        if a_new[i] < 0:
            a_new[i] = 0.0

    return a_new

######################################################################################################################################################
### FC, COV and COVtau calculation from Tims Series ###
def FC_COV_COVtau_from_ts(tsdata, NPARCELS, Tau=1):
    """
    Calculates FC, COV, and COVtau for each subject
    
    Parameters:
        tsdata: Empirical Time Series data (3D array: [NSUB, NPARCELS, timepoints] or 2D array: [NPARCELS, timepoints])
        NPARCELS: Number of parcels
        Tau: Time lag
    Returns:
        FCsub: Subject-specific functional connectivity matrix
        COVsub: Subject-specific covariance matrix
        COVtausub: Empirical functional connectivity matrix
    """
    indexN = np.arange(NPARCELS)  # Cortical areas
    N = len(indexN)

    NSUB = tsdata.shape[0]
    if tsdata.ndim == 2:
        tsdata = np.expand_dims(tsdata, axis=0)
        NSUB = 1

    FCsub = np.zeros((NSUB, NPARCELS, NPARCELS))
    COVsub = np.zeros((NSUB, NPARCELS, NPARCELS))
    COVtausub = np.zeros((NSUB, NPARCELS, NPARCELS))        
    for sub in range(NSUB):
        ts = tsdata[sub, :, :].copy()    # Extract subject's time series
        ts2 = ts[indexN, 10:-10].copy()  # Remove edges

        # Empirical FC(0)
        FC = np.corrcoef(ts2)
        FCsub[sub] = FC.copy()
        COV = np.cov(ts2)
        COVsub[sub] = COV.copy()

        # Empirical COV(tau)
        tst = ts2.T
        COVtau = np.zeros((N, N))
        sigratio = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                sigratio[i, j] = 1 / np.sqrt(COV[i, i]) / np.sqrt(COV[j, j])
                clag, lags = xcov(tst[:, i], tst[:, j], Tau, norm='none') #OK
                indx = np.where(lags == Tau)[0][0]
                COVtau[i, j] = clag[indx] / tst.shape[0]
        COVtau *= sigratio
        COVtausub[sub] = COVtau.copy()

    return FCsub, COVsub, COVtausub

######################################################################################################################################################
### Linear Hopf integration ###
def hopf_int(gC, f_diff, sigma, a):
    """
    Computes the linearized Hopf model.
    
    Parameters:
        gC: Connectivity matrix
        f_diff: Frequency differences
        sigma: Noise variance
        a: bifurcation parameter (vectorized)
    Returns:
        FC: Functional connectivity matrix
        CV: Covariance matrix
        Cvth: Full covariance matrix
        A: Jacobian matrix
    """
    N = gC.shape[0]
    wo = f_diff * (2 * np.pi)

    Cvth = np.zeros((2 * N, 2 * N))

    # Jacobian
    s = np.sum(gC, axis=1)
    B = np.diag(s)

    Axx = np.diag(a) - B + gC
    Ayy = Axx.copy()
    Ayx = np.diag(wo)
    Axy = -Ayx.copy()

    A = np.block([[Axx, Axy], [Ayx, Ayy]])

    # Noise covariance matrix
    if np.isscalar(sigma):  # Homogeneous noise
        Qn = np.diag([sigma**2] * (2 * N))
    else:  # Heterogeneous noise
        Qn = np.diag(np.concatenate([sigma**2, sigma**2]))

    # Solve Sylvester equation
    Cvth = solve_sylvester(A, A.T, -Qn)

    # Correlation from covariance
    FCth = corrcov_py_numba(Cvth)
    FC = FCth[:N, :N].copy()
    CV = Cvth[:N, :N].copy()

    return FC, CV, Cvth, A

##################################################
### Correlation matrix from covariance matrix ###
def corrcov_py(C):
    """
    Compute the correlation matrix from a covariance matrix.
    Equivalent to MATLAB's corrcov.
    """
    d = np.sqrt(np.diag(C))
    corr = C / d[:, None] / d[None, :]
    # Replace any NaNs resulting from zero-variance
    corr[np.isnan(corr)] = 0.0
    return corr

def corrcov(cov, ensure_symmetry=True, replace_nans=True):
    """
    Convert a covariance matrix into a correlation matrix.

    Parameters
    ----------
    cov : ndarray
        A square, symmetric, positive semi-definite covariance matrix.
    ensure_symmetry : bool, optional
        If True, symmetrizes the input by averaging with its transpose.
    replace_nans : bool, optional
        If True, replaces NaNs (from division by zero) with 0.

    Returns
    -------
    corr : ndarray
        The correlation matrix.
    """
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("Input must be a square 2D covariance matrix.")

    if ensure_symmetry:
        cov = (cov + cov.T) / 2

    d = np.sqrt(np.diag(cov))
    outer = np.outer(d, d)

    with np.errstate(divide='ignore', invalid='ignore'):
        corr = cov / outer

    if replace_nans:
        corr[np.isnan(corr)] = 0.0

    return corr

@njit
def corrcov_py_numba(C):
    N = C.shape[0]
    corr = np.empty((N, N))
    stddev = np.empty(N)

    for i in range(N):
        stddev[i] = np.sqrt(C[i, i]) if C[i, i] > 0 else 0.0

    for i in range(N):
        for j in range(N):
            denom = stddev[i] * stddev[j]
            if denom == 0.0:
                corr[i, j] = 0.0
            else:
                val = C[i, j] / denom
                corr[i, j] = val if np.isfinite(val) else 0.0

    return corr

##################################################
### Implementation of Matlab's xcov ###
import numpy as np
from scipy.signal import correlate

def xcov(x, y=None, Tau=None, norm='none'):
    """
    Replicate MATLAB's xcov behavior in Python.

    Parameters:
        x: First input signal (1D array).
        y: Second input signal (1D array). If None, auto-covariance of x is computed.
        Tau: Maximum lag (integer). If None, full range is returned.
        norm: Normalization option ('none', 'coeff').

    Returns:
        clag: Cross-covariance values.
        lags: Corresponding lags.
    """
    x = np.asarray(x)
    if y is None:
        y = x
    else:
        y = np.asarray(y)

    # Remove mean
    x = x - np.mean(x)
    y = y - np.mean(y)

    # Cross-correlation
    full_corr = correlate(x, y, mode='full')
    lags = np.arange(-len(x) + 1, len(y))

    # Lag trimming
    if Tau is not None:
        valid = (lags >= -Tau) & (lags <= Tau)
        full_corr = full_corr[valid]
        lags = lags[valid]

    # Normalization
    if norm == 'coeff':
        denom = np.sqrt(np.dot(x, x) * np.dot(y, y))
        full_corr = full_corr / denom if denom != 0 else full_corr

    return full_corr, lags


##################################################
### Matrix exponential exp(Gamma) ###
from numba import jit #, njit, prange
@jit(nopython=True)
def exp_scaling_squaring(Gamma, m=10):
    """
    Compute exp(Gamma) using the Scaling and Squaring Method.
    
    Parameters:
        Gamma (ndarray): Input square matrix (NxN).
        m (int): Degree of the Pade approximant for the series expansion.
    
    Returns:
        ndarray: Matrix exponential exp(Gamma).
    """
    # Compute the norm of Gamma (1-norm)
    norm_Gamma = np.linalg.norm(Gamma, ord=1)
    
    # Scaling step: Find scaling factor 2^k
    k = max(0, int(np.ceil(np.log2(norm_Gamma))))  # Scale such that norm(Gamma / 2^k) is small
    Gamma_scaled = Gamma / (2**k)
    
    # Compute exp(Gamma_scaled) using a series expansion or Pade approximant
    N = Gamma.shape[0]
    result = np.eye(N)  # Initialize result as the identity matrix
    term = np.eye(N)    # Initialize current term as the identity matrix
    
    for i in range(1, m + 1):
        term = np.dot(term, Gamma_scaled) / i  # Compute the next term in the series
        result += term  # Add to the result
    
    # Squaring step: Square the result k times
    for _ in range(k):
        result = np.dot(result, result)
    
    return result
