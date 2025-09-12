import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm, solve_sylvester
from scipy.signal import correlate
from numba import njit, prange, jit

def LinHopf_Ceff_sigma_a_fitting_adam(tsdata, C, NPARCELS, TR, f_diff, sigma, a=-0.02, Tau=1,
                                     fit_Ceff=True, competitive_coupling=False,
                                     fit_sigma=True, sigma_reset=False,
                                     fit_a=True,
                                     learning_rate_Ceff=1e-3, learning_rate_sigma=1e-4, learning_rate_a=1e-5,
                                     beta1=0.9, beta2=0.999, epsilon=1e-8,
                                     MAXiter=10000, error_tol=5e-4, patience=3,
                                     Ceff_norm=True, maxC=0.2,
                                     iter_check=50, plot_evol=False, plot_evol_last=False):
    """
    Fits the "averaged" G.Cij matrix AND the sigma values using the FC with Adam optimizer.
    
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
        fit_a: Flag for fitting a
        learning_rate_Ceff: Learning rate for Ceff (Adam)
        learning_rate_sigma: Learning rate for sigma (Adam)
        learning_rate_a: Learning rate for a (Adam)
        beta1: Adam parameter for first moment
        beta2: Adam parameter for second moment
        epsilon: Adam parameter for numerical stability
        MAXiter: Maximum number of iterations
        error_tol: Tolerance for convergence
        patience: Number cycles before stopping
        Ceff_norm: Flag for normalizing the Ceff
        maxC: Normalization factor for Ceff
        iter_check: Number of iterations to check error
        plot_evol: Flag for plotting evolution of error

    Returns:
        Ceff_fit: Effective connectivity matrix
        sigma_fit: Standard deviation of noise
        a_fit: Bifurcation parameters
        FCemp: Empirical functional connectivity matrix
        FCsim: Simulated functional connectivity matrix
        error_iter: List of errors at every iter_check iterations
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
        
    # Compute empirical statistics
    for sub in range(NSUB):
        ts = tsdata[sub, :, :].copy()    # Extract subject's time series
        ts2 = ts[indexN, 5:-5].copy()  # Remove edges

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
                clag, lags = xcov(tst[:, i], tst[:, j], Tau, norm='none')
                indx = np.where(lags == Tau)[0][0]
                COVtauemp[i, j] = clag[indx] / tst.shape[0]
        COVtauemp *= sigratio
        COVtauPB.append(COVtauemp)

    ## Group FC, COV, and COVtau
    FCemp = np.mean(FCPB, axis=0)
    COVemp = np.mean(COVPB, axis=0)
    COVtauemp = np.mean(COVtauPB, axis=0)
    COVemp_diag = np.diag(COVemp)

    ### Initialize parameters ###
    Ceff_new = C.copy()
    Ceff_previous = Ceff_new.copy()
    sigma_ini = sigma * np.ones(NPARCELS)
    sigma_new = sigma_ini.copy()
    sigma_previous = sigma_new.copy()
    a_ini = a * np.ones(NPARCELS)
    a_new = a_ini.copy()
    a_previous = a_new.copy()

    # Initialize Adam optimizer variables
    # For Ceff
    m_Ceff = np.zeros_like(Ceff_new)  # First moment
    v_Ceff = np.zeros_like(Ceff_new)  # Second moment
    
    # For sigma
    m_sigma = np.zeros_like(sigma_new)  # First moment
    v_sigma = np.zeros_like(sigma_new)  # Second moment
    
    # For a
    m_a = np.zeros_like(a_new)  # First moment
    v_a = np.zeros_like(a_new)  # Second moment

    error_old = 1e5
    error_tol_old = 1e5
    errorFC_iter = []
    errorCOVtau_iter = []
    error_iter = []
    patience_counter = 0
    best_error = 1e5
    no_improvement_count = 0

    for iter in range(1, MAXiter + 1):
        ### Check for non-finite values ###
        if not np.all(np.isfinite(a_new)):
            print(f"Iter {iter:4d}/{MAXiter} ; error: {error_old:.7f} ; error_tol: {error_tol_old:.7f} | A not finite --> EXIT (return previous)")
            Ceff_fit = Ceff_previous
            sigma_fit = sigma_previous
            a_fit = a_previous
            break
            
        ### Linear Hopf FC ###
        FCsim, COVsim, COVsimtotal, A = hopf_int(Ceff_new, f_diff, sigma_new, a_new)

        COVtausim = (exp_scaling_squaring((Tau * TR) * A) @ COVsimtotal)[:N, :N].copy()
        
        sigratiosim = compute_sigratio_from_cov(COVsim)
        COVtausim *= sigratiosim

        # Check for errors every iter_check iterations
        if iter % iter_check == 0:
            errorFC_now = np.mean((FCemp - FCsim) ** 2)
            errorFC_iter.append(errorFC_now)
            errorCOVtau_now = np.mean((COVtauemp - COVtausim) ** 2)
            errorCOVtau_iter.append(errorCOVtau_now)

            error_now = errorFC_now + errorCOVtau_now
            error_iter.append(error_now)

            # Early stopping if error increases
            # if error_old < error_now:
            #     Ceff_fit = Ceff_previous
            #     sigma_fit = sigma_previous
            #     a_fit = a_previous
            #     break
                
            # # Convergence check
            # error_tol_now = (error_old - error_now) / error_now if error_now > 0 else 0
            
            # Check for improvement
            if error_now < best_error:
                best_error = error_now
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            # # Convergence criteria
            # if error_tol_now < error_tol or np.abs(error_tol_old - error_tol_now) < error_tol/10:
            #     patience_counter += 1
            #     if patience_counter >= patience:
            #         Ceff_fit = Ceff_new
            #         sigma_fit = sigma_new
            #         a_fit = a_new
            #         break
            # else:
            #     patience_counter = 0
                
            # Additional stopping if no improvement for too long
            if no_improvement_count > patience:
                print(f"No improvement for {no_improvement_count} checks, stopping early")
                Ceff_fit = Ceff_new
                sigma_fit = sigma_new
                a_fit = a_new
                break
                
            error_old = error_now
            error_tol_old = error_tol_now

            if plot_evol:
                _plot_evolution(error_iter, errorFC_iter, errorCOVtau_iter, 
                              sigma_previous, sigma_new, iter_check, N,
                              learning_rate_Ceff, learning_rate_sigma)

        ### Compute gradients ###
        grad_Ceff_FC, grad_Ceff_COVtau = compute_Ceff_gradients(C, FCemp, FCsim, COVtauemp, COVtausim)
        grad_sigma_FC, grad_sigma_COVtau = compute_sigma_gradients(FCemp, FCsim, COVtauemp, COVtausim, N)
        grad_a_FC, grad_a_COVtau = compute_a_gradients(FCemp, FCsim, COVtauemp, COVtausim, N)

        ### Adam updates ###
        if fit_Ceff:
            Ceff_previous = Ceff_new.copy()
            grad_Ceff = grad_Ceff_FC + grad_Ceff_COVtau
            Ceff_new, m_Ceff, v_Ceff = adam_update(Ceff_new, grad_Ceff, m_Ceff, v_Ceff,
                                                   learning_rate_Ceff, beta1, beta2, epsilon, iter)
            
            # Apply constraints
            if not competitive_coupling:
                Ceff_new = np.where((C > 0) | (np.arange(N)[:, None] == N - 1 - np.arange(N)), 
                                   np.maximum(Ceff_new, 0), 0)
            else:
                Ceff_new = np.where((C > 0) | (np.arange(N)[:, None] == N - 1 - np.arange(N)), 
                                   Ceff_new, 0)
            
            # Handle NaN values
            Ceff_new = np.where(np.isfinite(Ceff_new), Ceff_new, 0.0)
            
            # Normalize if requested
            if Ceff_norm:
                max_val = np.max(np.abs(Ceff_new))
                if max_val > 0:
                    Ceff_new *= maxC / max_val

        if fit_sigma:
            sigma_previous = sigma_new.copy()
            grad_sigma = grad_sigma_FC + grad_sigma_COVtau
            sigma_new, m_sigma, v_sigma = adam_update(sigma_new, grad_sigma, m_sigma, v_sigma,
                                                     learning_rate_sigma, beta1, beta2, epsilon, iter)
            
            # Apply constraints
            # if sigma_reset:
            #     sigma_new = np.where(sigma_new < 0, sigma_ini, sigma_new)
            # else:
            #     sigma_new = np.maximum(sigma_new, 0)
            
            ### sigma re-normalization ###
            # _, COVsim, _, _ = hopf_int(Ceff_new, f_diff, sigma_new, a_new)
            # COVsim_diag = np.diag(COVsim)
            # normalization_factor = np.sum(COVemp_diag) / np.sum(COVsim_diag)
            # sigma_new *= np.sqrt(normalization_factor)
        
        if fit_a:
            a_previous = a_new.copy()
            grad_a = grad_a_FC + grad_a_COVtau
            a_new, m_a, v_a = adam_update(a_new, grad_a, m_a, v_a,
                                         learning_rate_a, beta1, beta2, epsilon, iter)
            
            # Apply constraints (clip to reasonable range)
            a_new = np.clip(a_new, -0.1, -0.001)

    if iter == MAXiter:
        print('Reached max. iterations:', MAXiter)
        Ceff_fit = Ceff_new
        sigma_fit = sigma_new
        a_fit = a_new

    # Final simulation
    FCsim, _, _, _ = hopf_int(Ceff_fit, f_diff, sigma_fit, a_fit)

    if plot_evol_last:
        _plot_evolution(error_iter, errorFC_iter, errorCOVtau_iter, 
                      sigma_previous, sigma_new, iter_check, N,
                      learning_rate_Ceff, learning_rate_sigma)

    return Ceff_fit, sigma_fit, a_fit, FCemp, FCsim, error_iter, errorFC_iter, errorCOVtau_iter


@njit
def adam_update(param, grad, m, v, lr, beta1, beta2, eps, t):
    """
    Adam optimizer update step (numba-compiled for speed)
    
    Parameters:
        param: Current parameter values
        grad: Gradients
        m: First moment estimate
        v: Second moment estimate
        lr: Learning rate
        beta1: Exponential decay rate for first moment
        beta2: Exponential decay rate for second moment
        eps: Small constant for numerical stability
        t: Time step (iteration number)
    
    Returns:
        Updated parameter, first moment, second moment
    """
    # Update biased first moment estimate
    m_new = beta1 * m + (1 - beta1) * grad
    
    # Update biased second raw moment estimate
    v_new = beta2 * v + (1 - beta2) * grad * grad
    
    # Compute bias-corrected first moment estimate
    m_hat = m_new / (1 - beta1**t)
    
    # Compute bias-corrected second raw moment estimate
    v_hat = v_new / (1 - beta2**t)
    
    # Update parameters
    param_new = param - lr * m_hat / (np.sqrt(v_hat) + eps)
    
    return param_new, m_new, v_new


@njit(parallel=True)
def compute_Ceff_gradients(C, FCemp, FCsim, COVtauemp, COVtausim):
    """Compute gradients for Ceff matrix"""
    N = C.shape[0]
    grad_FC = np.zeros_like(C)
    grad_COVtau = np.zeros_like(C)
    
    for i in prange(N):
        for j in range(N):
            if C[i, j] > 0 or j == N - i - 1:
                grad_FC[i, j] = -(FCemp[i, j] - FCsim[i, j])
                grad_COVtau[i, j] = -(COVtauemp[i, j] - COVtausim[i, j])
    
    return grad_FC, grad_COVtau


@njit(parallel=True)
def compute_sigma_gradients(FCemp, FCsim, COVtauemp, COVtausim, N):
    """Compute gradients for sigma parameters"""
    grad_FC = np.zeros(N)
    grad_COVtau = np.zeros(N)
    
    for i in prange(N):
        grad_FC[i] = -np.sum(FCemp[i, :] - FCsim[i, :])
        grad_COVtau[i] = -np.sum(COVtauemp[i, :] - COVtausim[i, :])
    
    return grad_FC, grad_COVtau


@njit(parallel=True)
def compute_a_gradients(FCemp, FCsim, COVtauemp, COVtausim, N):
    """Compute gradients for a parameters"""
    grad_FC = np.zeros(N)
    grad_COVtau = np.zeros(N)
    
    for i in prange(N):
        grad_FC[i] = -np.sum(FCemp[i, :] - FCsim[i, :])
        grad_COVtau[i] = -np.sum(COVtauemp[i, :] - COVtausim[i, :])
    
    return grad_FC, grad_COVtau


def _plot_evolution(error_iter, errorFC_iter, errorCOVtau_iter, 
                   sigma_previous, sigma_new, iter_check, N,
                   lr_Ceff, lr_sigma):
    """Helper function for plotting evolution"""
    try:
        from IPython.display import clear_output
        clear_output(wait=True)
    except ImportError:
        pass
        
    plt.figure(figsize=(18, 5))
    plt.subplot(141)
    plt.title('error_iter')
    plt.plot(np.arange(1, len(error_iter) + 1) * iter_check, error_iter, 'o-')
    plt.plot(np.arange(1, len(error_iter) + 1) * iter_check, errorFC_iter, '.-', label='errorFC_iter')
    plt.plot(np.arange(1, len(error_iter) + 1) * iter_check, errorCOVtau_iter, '.-', label='errorCOVtau_iter')
    plt.xlabel('Iteration')
    plt.legend()
    
    plt.subplot(142)
    plt.title('errorFC_iter')
    plt.plot(np.arange(1, len(errorFC_iter) + 1) * iter_check, errorFC_iter, 'o-', color='tab:orange', 
             label=f'lr_Ceff={lr_Ceff:.3e}')
    plt.xlabel('Iteration')
    plt.legend()
    
    plt.subplot(143)
    plt.title('errorCOVtau_iter')
    plt.plot(np.arange(1, len(errorCOVtau_iter) + 1) * iter_check, errorCOVtau_iter, 'o-', color='tab:green', 
             label=f'lr_sigma={lr_sigma:.3e}')
    plt.xlabel('Iteration')
    plt.legend()
    
    plt.subplot(144)
    plt.title('sigma')
    plt.plot(np.arange(1, N + 1), sigma_previous, 'o-', label='previous')
    plt.plot(np.arange(1, N + 1), sigma_new, '.-', label='new')
    plt.xlabel('Parcel')
    plt.legend()
    plt.show()


# Keep all the original helper functions
@njit
def compute_sigratio_from_cov(COV):
    N = COV.shape[0]
    sigratio = np.zeros((N, N))
    for i in range(N):
        sqrt_ii = np.sqrt(COV[i, i])
        for j in range(N):
            sigratio[i, j] = 1.0 / (sqrt_ii * np.sqrt(COV[j, j]))
    return sigratio


def hopf_int(gC, f_diff, sigma, a):
    """
    Computes the linearized Hopf model.
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


def xcov(x, y=None, Tau=None, norm='none'):
    """
    Replicate MATLAB's xcov behavior in Python.
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


@jit(nopython=True)
def exp_scaling_squaring(Gamma, m=10):
    """
    Compute exp(Gamma) using the Scaling and Squaring Method.
    """
    # Compute the norm of Gamma (1-norm)
    norm_Gamma = np.linalg.norm(Gamma, ord=1)
    
    # Scaling step: Find scaling factor 2^k
    k = max(0, int(np.ceil(np.log2(norm_Gamma))))
    Gamma_scaled = Gamma / (2**k)
    
    # Compute exp(Gamma_scaled) using a series expansion
    N = Gamma.shape[0]
    result = np.eye(N)
    term = np.eye(N)
    
    for i in range(1, m + 1):
        term = np.dot(term, Gamma_scaled) / i
        result += term
    
    # Squaring step: Square the result k times
    for _ in range(k):
        result = np.dot(result, result)
    
    return result


# Keep all other original functions (FC_COV_COVtau_from_ts, corrcov, etc.)
def FC_COV_COVtau_from_ts(tsdata, NPARCELS, Tau=1):
    """
    Calculates FC, COV, and COVtau for each subject
    """
    indexN = np.arange(NPARCELS)
    N = len(indexN)

    NSUB = tsdata.shape[0]
    if tsdata.ndim == 2:
        tsdata = np.expand_dims(tsdata, axis=0)
        NSUB = 1

    FCsub = np.zeros((NSUB, NPARCELS, NPARCELS))
    COVsub = np.zeros((NSUB, NPARCELS, NPARCELS))
    COVtausub = np.zeros((NSUB, NPARCELS, NPARCELS))        
    for sub in range(NSUB):
        ts = tsdata[sub, :, :].copy()
        ts2 = ts[indexN, 10:-10].copy()

        FC = np.corrcoef(ts2)
        FCsub[sub] = FC.copy()
        COV = np.cov(ts2)
        COVsub[sub] = COV.copy()

        tst = ts2.T
        COVtau = np.zeros((N, N))
        sigratio = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                sigratio[i, j] = 1 / np.sqrt(COV[i, i]) / np.sqrt(COV[j, j])
                clag, lags = xcov(tst[:, i], tst[:, j], Tau, norm='none')
                indx = np.where(lags == Tau)[0][0]
                COVtau[i, j] = clag[indx] / tst.shape[0]
        COVtau *= sigratio
        COVtausub[sub] = COVtau.copy()

    return FCsub, COVsub, COVtausub