import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
#
from numba import jit, njit, prange

###############################################################
### Simulation of the Linear Langevin (also Linear-Hopf) ######

@jit(nopython=True)
def Integrate_Langevin_ND_Optimized(Gamma, sigma, initcond, duration, integstep):
    """
    Integrate a zero-order Langevin equation with Numba optimization
    dx/dt = - Gamma * x + noise

    Parameters
    ----------
    Gamma : NxN matrix
    sigma : scalar or Nx1 vector
        Standard deviation of the external noise
    initcond : Nx1 vector 
        Initial condition x(0)
    duration : scalar
        Length of the simulation in arbitrary units
    integstep : scalar
        Integration time step per simulation unit time
    
    Returns
    -------
    X : ndarray (N,timesteps)
        Integration of the first-order Langevin equation
    noise : ndarray (N,timesteps)
        Generated noise array
    """
    Ndim = Gamma.shape[0]
    nsteps = int(duration / integstep) + 1

    X = np.zeros((Ndim, nsteps))
    X[:, 0] = initcond

    # Generate noise
    noise = np.zeros((Ndim, nsteps))
    sigma_array = sigma * np.ones(Ndim)
    noise = sigma_array[:, np.newaxis] * np.random.standard_normal((Ndim, nsteps))
    noisedt = np.sqrt(integstep) * noise

    for t in range(1, nsteps):
        for i in range(Ndim):
            # Compute the damping term (Gamma @ X[:, t-1])
            damping = 0.0
            for j in range(Ndim):
                damping += Gamma[i, j] * X[j, t-1]
            
            # Update X using the explicit formula
            X[i, t] = (
                X[i, t-1]
                - integstep * damping  # Explicitly subtract damping
                + noisedt[i, t]        # Add noise
            )

    return X, noise

###############################################################
### Construct A matrix for Linear Hopf ########################

def construct_matrix_A(a, omega, C, g):
    # Ensure inputs are numpy arrays
    a = np.array(a)
    omega = np.array(omega)
    C = np.array(C)
    # Calculate S
    S = np.sum(C, axis=1)
    # Diagonal matrices
    diag_a_minus_gS = np.diag(a - g * S)
    diag_omega = np.diag(omega)
    # Matrices A_xx, A_yy
    A_xx = diag_a_minus_gS + g * C
    A_yy = A_xx
    # Matrices A_xy, A_yx
    A_xy = diag_omega
    A_yx = -diag_omega
    # Construct the full matrix A
    A = np.block([
        [A_xx, A_xy],
        [A_yx, A_yy]
    ])
    return A

###############################################################
### Simulation of the Hopf model ##############################

@jit(nopython=True)
def Integrate_Hopf_ND_Optimized(BIFURCATION_PARAM_a, NODES_FREQS_w, CONNECTIVITY_C, GLOBAL_COUPLING_G, NOISE_SIGMA, X0, Y0, DURATION, INTEGSTEP):
    """
    Integrate the Hopf model with Numba optimization
    
    Parameters
    ----------
    BIFURCATION_PARAM_a : scalar or Nx1 vector
        Bifurcation parameter
    NODES_FREQS_w : Nx1 vector
        Nodes frequencies 
    CONNECTIVITY_C : NxN
        Connectivity or GEC matrix
    GLOBAL_COUPLING_G : scalar
        Global coupling parameter
    NOISE_SIGMA : scalar or Nx1 vector
        Standard deviation of the external noise
    X0 : Nx1 vector
        Initial condition x(0)
    Y0 : Nx1 vector
        Initial condition y(0)
    DURATION : scalar
        Length of the simulation in arbitrary units
    INTEGSTEP : scalar
        Integration time step per simulation unit time
    
    Returns
    -------
    X : ndarray (N,timesteps)
        Simulated signal
    Y : ndarray (N,timesteps)
        Simulated signal (Imaginary part, used for intial condition after thermalizing)
    noiseX_sim : ndarray (N,timesteps)
        Generated noise array for the X component
    """
    nparcels = CONNECTIVITY_C.shape[0]
    nsteps = int(DURATION / INTEGSTEP) + 1
    
    a_array = BIFURCATION_PARAM_a * np.ones(nparcels)
    sigma_array = NOISE_SIGMA * np.ones(nparcels)

    # Normalize and scale the structural connectivity
    max_SC = np.max(CONNECTIVITY_C)
    if max_SC == 0:
        max_SC = 1
    normalized_SC = CONNECTIVITY_C / max_SC
    weak_SC = 0.2 * normalized_SC
    weighted_conn = GLOBAL_COUPLING_G * weak_SC

    # Create the other term that we need for the implentation
    # Sum. Connectivity
    sum_conn = np.empty((weighted_conn.shape[0], 2), dtype=weighted_conn.dtype)
    sum_conn[:, 0] = weighted_conn.sum(1)
    sum_conn[:, 1] = weighted_conn.sum(1)
    # Bifurcation parameter
    a_hopf = np.empty((a_array.shape[0], 2), dtype=a_array.dtype)
    a_hopf[:, 0] = a_array
    a_hopf[:, 1] = a_array
    # Nodes Frequencies
    omega = np.zeros((nparcels, 2))
    omega[:, 0] = - NODES_FREQS_w
    omega[:, 1] =   NODES_FREQS_w

    # Initial conditions
    z = np.column_stack((X0, Y0))   # x = z[:, 0], y = z[:, 1]
    X = np.zeros((nparcels, nsteps))
    Y = np.zeros((nparcels, nsteps))

    # Generate noise
    noiseX = sigma_array[:, np.newaxis] * np.random.standard_normal((nparcels, nsteps))
    noiseY = sigma_array[:, np.newaxis] * np.random.standard_normal((nparcels, nsteps))

    # Keep the transients
    for t in range(0, nsteps):
        zz = z[:, ::-1]  # flipped so that zz[:, 0] = y; zz[:, 1] = x
        interaction = weighted_conn @ z - sum_conn * z  # sum(Cij*xi) - sum(Cij)*xj
        bifur_freq = a_hopf * z + zz * omega  # Bifurcation factor and freq terms
        intra_terms = z * (z*z + zz*zz)
        
        noise = np.column_stack((noiseX[:,t], noiseY[:,t]))
        noisedt = np.sqrt(INTEGSTEP) * noise
        
        # Integrative step
        X[:, t] = z[:, 0]
        Y[:, t] = z[:, 1]
        # noiseX_sim[:, t] = noiseX
        z = z + INTEGSTEP * (bifur_freq - intra_terms + interaction) + noisedt

    return X, Y, noiseX

#############################
@jit(nopython=True)
def Integrate_Hopf_ND_Optimized_2(BIFURCATION_PARAM_a, NODES_FREQS_w, CONNECTIVITY_C, GLOBAL_COUPLING_G, NOISE_SIGMA, X0, Y0, DURATION, INTEGSTEP):
    """
    Simulates the Hopf model starting from an initialized state z0.
    """
    nparcels = CONNECTIVITY_C.shape[0]

    a_array = BIFURCATION_PARAM_a * np.ones(nparcels)
    a_hopf = np.column_stack((a_array,a_array))
    omega = np.column_stack(( - NODES_FREQS_w, NODES_FREQS_w))
    sigma_array = NOISE_SIGMA * np.ones(nparcels)

    # Normalize and scale the structural connectivity
    max_SC = np.max(CONNECTIVITY_C)
    if max_SC == 0:
        max_SC = 1
    wC = GLOBAL_COUPLING_G * CONNECTIVITY_C / max_SC * 0.2
    row_sums = np.sum(wC, axis=1).reshape(-1, 1)
    sumC = np.column_stack((row_sums, row_sums))
    z0 = np.column_stack((X0, Y0))

    # Initialize variables
    Tmax = DURATION
    dt = INTEGSTEP
    nsteps = int(Tmax / dt) + 1
    z = z0.copy()
    xs = np.zeros((nsteps, nparcels))
    ys = np.zeros((nsteps, nparcels))
    force = np.zeros((nsteps, nparcels))
    noise = np.zeros((nsteps, nparcels))

    # Simulation loop
    for t in range(0, nsteps):
        # Coupling and flipped z
        suma = wC @ z - sumC * z
        zz = z[:, ::-1]  # Flip columns (x and y)

        # Hopf terms
        cubic_term = -z * (z**2 + zz**2)
        fo = a_hopf * z + zz * omega + cubic_term

        # Noise term
        noiseX = sigma_array * np.random.standard_normal(nparcels)
        noiseY = sigma_array * np.random.standard_normal(nparcels)
        no = np.column_stack((noiseX,noiseY))

        # Langevin equation update
        z = z + dt * (fo + suma) + np.sqrt(dt) * no

        # Save simulated time series, force, and noise
        xs[t, :] = z[:, 0]
        ys[t, :] = z[:, 1]
        force[t, :] = (fo[:, 0] + suma[:, 0])
        noise[t, :] = no[:, 0]

    # Post-processing
    ts_x = xs.T
    ts_y = ys.T
    force_x = -force.T  # Consistent with Cugliandolo 1994
    noise_x = (noise.T)  # Normalize noise by sqrt(dt)

    return ts_x, ts_y, force_x, noise_x

###############################################################
### Non-Eq FDT Calculation ####################################

# @jit(nopython=True)
@njit(parallel=True)
def FDT_Langevin_ND_Optimized(nsim, Gamma, sigma, v0std, v0bias, Thermalization, duration, integstep):
    Ndim = np.shape(Gamma)[0]
    nsteps = int(duration / integstep) + 1

    Cts = np.zeros((Ndim, nsteps, nsteps))
    Rts = np.zeros((Ndim, nsteps, nsteps))
    Ats = np.zeros((Ndim, nsteps, nsteps))
    V_0 = np.zeros_like(Gamma)
    Gamma = np.ascontiguousarray(Gamma)  # Make Gamma contiguous

    inv_nsim = 1.0 / nsim
    inv_nsim_sigma2_sqrtdt = np.ones(Ndim) / (nsim * sigma**2 * np.sqrt(integstep))

    for i in range(nsim):
        # Set initial conditions
        initcond = v0std * np.random.standard_normal(Ndim) + v0bias
        if Thermalization == 1:
            # Discard first "tfinal/dt" time-points
            term_time = 100
            dt = integstep
            v0 = initcond
            vsim, _ = Integrate_Langevin_ND_Optimized(Gamma, sigma, initcond=v0, duration=term_time, integstep=dt)
            initcond = vsim[:, -1]
        V_0 += np.outer(initcond,initcond)

        # Integrate Langevin dynamics
        Xsim, noise = Integrate_Langevin_ND_Optimized(Gamma, sigma, initcond, duration, integstep)

        for i in prange(Ndim):
            for t in range(nsteps):
                for s in range(0,t):
                    Xsim_it = Xsim[i, t]
                    Xsim_is = Xsim[i, s]
                    
                    Cts[i, t, s] += Xsim_it * Xsim_is
                    Rts[i, t, s] += Xsim_it * noise[i, s]
                    
                    for j in range(Ndim):
                        Gamma_ij = Gamma[i, j]
                        Xsim_jt = Xsim[j, t]
                        Xsim_js = Xsim[j, s]
                        
                        Ats[i, t, s] += (Gamma_ij * Xsim_jt * Xsim_is - Gamma_ij * Xsim_js * Xsim_it)
                    
    # # Ensemble average
    Cts *= inv_nsim
    for i in prange(Ndim):
        Rts[i,:,:] *= inv_nsim_sigma2_sqrtdt[i]
    Ats *= inv_nsim
    V_0 *= inv_nsim
    return Cts, Rts, Ats, V_0


@jit(nopython=True)
def FDT_Hopf_ND_Optimized(nsim, BIFURCATION_PARAM_a, NODES_FREQS_w, CONNECTIVITY_C, GLOBAL_COUPLING_G, NOISE_SIGMA, v0std, v0bias, Thermalization, duration, integstep):
    Ndim = np.shape(CONNECTIVITY_C)[0]
    nsteps = int(duration / integstep) + 1

    Cts = np.zeros((Ndim, nsteps, nsteps))
    Rts = np.zeros((Ndim, nsteps, nsteps))
    # Ats = np.zeros((Ndim, nsteps, nsteps))
    V_0 = np.zeros_like(CONNECTIVITY_C)
    CONNECTIVITY_C = np.ascontiguousarray(CONNECTIVITY_C)  # Make Connectivity contiguous

    inv_nsim = 1.0 / nsim
    inv_nsim_sigma2_sqrtdt = np.ones(Ndim) / (nsim * NOISE_SIGMA**2 * np.sqrt(integstep))

    # For the other Hopf
    wC = CONNECTIVITY_C
    row_sums = np.sum(wC, axis=1).reshape(-1, 1)
    sumC = np.column_stack((row_sums, row_sums))
    omega_xy = np.column_stack((NODES_FREQS_w, NODES_FREQS_w))

    for i in range(nsim):
        # Set initial conditions
        X0 = v0std * np.random.standard_normal(Ndim) + v0bias
        Y0 = v0std * np.random.standard_normal(Ndim) + v0bias
        if Thermalization == 1:
            # Discard first "tfinal/dt" time-points
            term_time = 100
            dt = integstep
            Xsim, Ysim, _ = Integrate_Hopf_ND_Optimized(BIFURCATION_PARAM_a, NODES_FREQS_w, CONNECTIVITY_C, GLOBAL_COUPLING_G, NOISE_SIGMA, X0, Y0, term_time, dt)
            
            # z0 = np.column_stack((X0, Y0))
            # Xsim, Ysim, _, _ = Integrate_Hopf_ND_Optimized_2(ahopf=BIFURCATION_PARAM_a, omega=omega_xy, sigma=NOISE_SIGMA, dt=dt, Tmax=term_time, wC=wC, sumC=sumC, z0=z0)

            X0 = Xsim[:, -1]
            Y0 = Ysim[:, -1]
        V_0 += np.outer(X0,X0)

        # Integrate Hopf model
        Xsim, _, noise = Integrate_Hopf_ND_Optimized(BIFURCATION_PARAM_a, NODES_FREQS_w, CONNECTIVITY_C, GLOBAL_COUPLING_G, NOISE_SIGMA, X0, Y0, duration, integstep)

        # z0 = np.column_stack((X0, Y0))
        # Xsim, _, _, noise = Integrate_Hopf_ND_Optimized_2(ahopf=BIFURCATION_PARAM_a, omega=omega_xy, sigma=NOISE_SIGMA, dt=integstep, Tmax=duration, wC=wC, sumC=sumC, z0=z0)

        Xsim = Xsim.T

        for i in prange(Ndim):
            for t in range(nsteps):
                for s in range(0,t):
                    Xsim_it = Xsim[i, t]
                    Xsim_is = Xsim[i, s]
                    
                    Cts[i, t, s] += Xsim_it * Xsim_is
                    Rts[i, t, s] += Xsim_it * noise[i, s]
                    
                    # for j in range(Ndim):
                    #     Gamma_ij = Gamma[i, j]
                    #     Xsim_jt = Xsim[j, t]
                    #     Xsim_js = Xsim[j, s]
                    #     Ats[i, t, s] += (Gamma_ij * Xsim_jt * Xsim_is - Gamma_ij * Xsim_js * Xsim_it)
                    
    # # Ensemble average
    Cts *= inv_nsim
    for i in prange(Ndim):
        Rts[i,:,:] *= inv_nsim_sigma2_sqrtdt[i]
    V_0 *= inv_nsim
    # Ats *= inv_nsim
    return Cts, Rts, V_0 #, Ats


###############################################################
### Analytic Expressions ######################################    

### V_0 = <v.v^T>
def V_0_calculation_v0gauss(v0std, v0bias, Ndim):
    I = np.ones(Ndim)
    v0bias_array = v0bias * I
    v0std_square_array = v0std**2 * I
    term1 = np.diag(v0std_square_array)
    term2 = np.zeros((Ndim,Ndim))
    for i in range(Ndim):
        for j in range(Ndim):
            term2[i,j] = v0bias_array[i] * v0bias_array[j]
    V_0 = term1 + term2
    return V_0

def V_0_calculation_v0fixed(v0):
    V_0 = np.outer(v0,v0)
    return V_0

### Compute Cts ###############################################
@jit(forceobj=True)
def Cts_Langevin_ND(Gamma, sigma, V_0, t, s):
    # Exponential matrices using scipy.linalg.expm
    exp_Gamma_t = exp_scaling_squaring(-Gamma * t)
    exp_GammaT_s = exp_scaling_squaring(-Gamma.T * s)
    # First term: (e^{-\Gamma t} V_0 e^{-\Gamma^T s})_{ii}
    term1 = np.diag(exp_Gamma_t @ V_0 @ exp_GammaT_s)
    # Second term: 
    term2 = compute_term2_of_C_precompute(t, s, Gamma, sigma**2)
    C = term1 + term2
    return C

### term2 numerical integration
from scipy.integrate import quad
@jit(forceobj=True)
def compute_term2_of_C_num_int(t, s, Gamma, sigma_squared):
    N = Gamma.shape[0]
    term2 = np.zeros(N)  # Initialize B as a column vector

    def integrand(tau, i):
        exp1 = exp_scaling_squaring(-Gamma * (t - tau))
        exp2 = exp_scaling_squaring(-Gamma.T * (s - tau))
        return np.sum(sigma_squared * exp1[i, :] * exp2[:, i])

    for i in range(N):
        term2[i], _ = quad(integrand, 0, s, args=(i,))
    
    return term2

### term2 analytic calculation
def compute_term2_of_C_precompute(t, s, Gamma, sigma_squared):
    """
    Wrapper to precompute eigenvalues and eigenvectors for a general matrix Gamma.
    """
    eigvals, eigvecs = np.linalg.eig(Gamma)
    P = eigvecs
    P_inv = np.linalg.inv(P)
    return compute_term2_of_C_analytic_numba(t, s, eigvals, P, P_inv, sigma_squared)

@njit
def compute_term2_of_C_analytic_numba(t, s, eigvals, P, P_inv, sigma_squared):
    """
    Optimized computation of the vector B(t, s) using Numba.
    """
    N = len(eigvals)
    
    # eigvals = np.real(eigvals)

    # Ensure sigma_squared is an array
    if np.isscalar(sigma_squared):
        sigma_squared = sigma_squared * np.ones(N)
    
    # Precompute sigma weight terms
    S = np.zeros((N, N), dtype=np.complex128)  # Explicitly complex
    for k in range(N):
        for m in range(N):
            S[k, m] = np.sum(sigma_squared * P_inv[k, :] * P_inv[m, :])
    
    # Precompute exponentials and integration terms
    integral_term = np.zeros((N, N), dtype=np.complex128)  # Explicitly complex
    for k in range(N):
        eigval_k = eigvals[k]
        for m in range(N):
            eigval_m = eigvals[m]
            denom = eigval_k + eigval_m
            if np.abs(denom) > 1e-12:  # Avoid division by zero
                integral_term[k, m] = (np.exp(-eigval_k * (t - s)) - np.exp(-(eigval_k * t + eigval_m * s))) / denom
            else:
                integral_term[k, m] = s * np.exp(-(eigval_k * t + eigval_m * s)) # Special case when denom is zero
    
    # Compute B[i]
    term2_of_C = np.zeros(N, dtype=np.complex128)
    for i in range(N):
        for k in range(N):
            for m in range(N):
                term_km = (
                    P[i, k] * P[i, m] *
                    integral_term[k, m] *
                    S[k, m]
                )
                term2_of_C[i] += term_km
    
    return np.real(term2_of_C)  # Return the real part

### Compute dC/dt ###############################################
@jit(forceobj=True)
def dtCts_Langevin_ND(Gamma, sigma, V_0, t, s):
    # Exponential matrices using scipy.linalg.expm
    exp_Gamma_t = exp_scaling_squaring(-Gamma * t)
    exp_GammaT_s = exp_scaling_squaring(-Gamma.T * s)
    # First term: (e^{-\Gamma t} V_0 e^{-\Gamma^T s})_{ii}
    term1 = (-1) * np.diag(Gamma @ exp_Gamma_t @ V_0 @ exp_GammaT_s)
    # Second term: 
    term2 = compute_term2_of_dtC_precompute(t, s, Gamma, sigma**2)
    C = term1 + term2
    return C

### term2 of dC/dt analytic calculation
def compute_term2_of_dtC_precompute(t, s, Gamma, sigma_squared):
    """
    Wrapper to precompute eigenvalues and eigenvectors for a general matrix Gamma.
    """
    eigvals, eigvecs = np.linalg.eig(Gamma)
    P = eigvecs
    P_inv = np.linalg.inv(P)
    return compute_term2_of_dtC_analytic_numba(t, s, eigvals, P, P_inv, sigma_squared)
@njit
def compute_term2_of_dtC_analytic_numba(t, s, eigvals, P, P_inv, sigma_squared):
    """
    Optimized computation of the vector B(t, s) using Numba.
    """
    N = len(eigvals)
    
    # eigvals = np.real(eigvals)

    # Ensure sigma_squared is an array
    if np.isscalar(sigma_squared):
        sigma_squared = sigma_squared * np.ones(N)
    
    # Precompute sigma weight terms
    S = np.zeros((N, N), dtype=np.complex128)  # Explicitly complex
    for k in range(N):
        for m in range(N):
            S[k, m] = np.sum(sigma_squared * P_inv[k, :] * P_inv[m, :])
    
    # Precompute exponentials and integration terms
    integral_term = np.zeros((N, N), dtype=np.complex128)  # Explicitly complex
    for k in range(N):
        eigval_k = eigvals[k]
        for m in range(N):
            eigval_m = eigvals[m]
            denom = eigval_k + eigval_m
            exp_kt_ms = np.exp(-(eigval_k * t + eigval_m * s))
            exp_kts = np.exp(-eigval_k * (t - s))
            if np.abs(denom) > 1e-12:  # Avoid division by zero
                integral_term[k, m] = -eigval_k * (exp_kts - exp_kt_ms) / denom
            else:
                integral_term[k, m] = -eigval_k * s * exp_kt_ms
    
    # Compute B[i]
    term2_of_dtC = np.zeros(N, dtype=np.complex128)
    for i in range(N):
        for k in range(N):
            for m in range(N):
                term_km = (
                    P[i, k] * P[i, m] *
                    integral_term[k, m] *
                    S[k, m]
                )
                term2_of_dtC[i] += term_km
    
    return np.real(term2_of_dtC)  # Return the real part

def dtCts_Langevin_ND_numerical(Gamma, sigma, V_0, t, s, h=1e-5):
    C_t_plus_h = Cts_Langevin_ND(Gamma, sigma, V_0, t + h, s)
    C_t_minus_h = Cts_Langevin_ND(Gamma, sigma, V_0, t - h, s)
    return (C_t_plus_h - C_t_minus_h) / (2 * h)

### Compute dC/ds ###############################################
@jit(forceobj=True)
def dsCts_Langevin_ND(Gamma, sigma, V_0, t, s):
    # Exponential matrices using scipy.linalg.expm
    exp_Gamma_t = exp_scaling_squaring(-Gamma * t)
    exp_GammaT_s = exp_scaling_squaring(-Gamma.T * s)
    # First term: (e^{-\Gamma t} V_0 e^{-\Gamma^T s})_{ii}
    term1 = (-1) * np.diag(exp_Gamma_t @ V_0 @ Gamma.T @ exp_GammaT_s)
    # Second term: 
    term2 = compute_term2_of_dsC_precompute(t, s, Gamma, sigma**2)
    C = term1 + term2
    return C

### term2 of dC/dt analytic calculation
def compute_term2_of_dsC_precompute(t, s, Gamma, sigma_squared):
    """
    Wrapper to precompute eigenvalues and eigenvectors for a general matrix Gamma.
    """
    eigvals, eigvecs = np.linalg.eig(Gamma)
    P = eigvecs
    P_inv = np.linalg.inv(P)
    return compute_term2_of_dsC_analytic_numba(t, s, eigvals, P, P_inv, sigma_squared)
@njit
def compute_term2_of_dsC_analytic_numba(t, s, eigvals, P, P_inv, sigma_squared):
    """
    Optimized computation of the vector B(t, s) using Numba.
    """
    N = len(eigvals)
    
    # eigvals = np.real(eigvals)

    # Ensure sigma_squared is an array
    if np.isscalar(sigma_squared):
        sigma_squared = sigma_squared * np.ones(N)
    
    # Precompute sigma weight terms
    S = np.zeros((N, N), dtype=np.complex128)  # Explicitly complex
    for k in range(N):
        for m in range(N):
            S[k, m] = np.sum(sigma_squared * P_inv[k, :] * P_inv[m, :])
    
    # Precompute exponentials and integration terms
    integral_term = np.zeros((N, N), dtype=np.complex128)  # Explicitly complex
    for k in range(N):
        eigval_k = eigvals[k]
        for m in range(N):
            eigval_m = eigvals[m]
            denom = eigval_k + eigval_m
            exp_kms = np.exp((eigval_k + eigval_m) * s)
            exp_kt_ms = np.exp(-(eigval_k * t + eigval_m * s))
            if np.abs(denom) > 1e-12:  # Avoid division by zero
                integral_term[k, m] = exp_kt_ms * (exp_kms - eigval_m * (exp_kms - 1) / denom)
            else:
                integral_term[k, m] = exp_kt_ms * (1 - eigval_m * s)
    
    # Compute B[i]
    term2_of_dsC = np.zeros(N, dtype=np.complex128)
    for i in range(N):
        for k in range(N):
            for m in range(N):
                term_km = (
                    P[i, k] * P[i, m] *
                    integral_term[k, m] *
                    S[k, m]
                )
                term2_of_dsC[i] += term_km
    
    return np.real(term2_of_dsC)  # Return the real part

def dsCts_Langevin_ND_numerical(Gamma, sigma, V_0, t, s, h=1e-5):
    C_s_plus_h = Cts_Langevin_ND(Gamma, sigma, V_0, t, s + h)
    C_s_minus_h = Cts_Langevin_ND(Gamma, sigma, V_0, t, s - h)
    return (C_s_plus_h - C_s_minus_h) / (2 * h)

### Compute R ###############################################
@jit(forceobj=True)
def Rts_Langevin_ND(Gamma, t, s):
    # Compute matrix exponential
    exp_Gamma_t_minus_s = exp_scaling_squaring(-Gamma * (t - s))
    # Extract the diagonal elements
    Rts = np.diag(exp_Gamma_t_minus_s)
    return Rts

### Compute int_R ############################################
@jit(forceobj=True)
def intRts_Langevin_ND(Gamma, t, s):
    # Compute matrix exponential
    exp_Gamma_t_minus_s = exp_scaling_squaring(-Gamma * (t - s))
    # Compute the identity matrix
    I = np.eye(Gamma.shape[0])
    intRts = np.diag(np.linalg.inv(Gamma) @ (I - exp_Gamma_t_minus_s))
    return intRts

### Compute int_R_norm1 = int_R_i/(Gamma^-1_ii) ############
def intRts_norm1_Langevin_ND(Gamma, t, s):
    # Matrix exponential
    exp_Gamma_t_minus_s = exp_scaling_squaring(-Gamma * (t - s))
    I = np.eye(Gamma.shape[0])
    Gamma_inv = np.linalg.inv(Gamma)
    # Numerator: diagonal of susceptibility matrix
    chi = np.diag(Gamma_inv @ (I - exp_Gamma_t_minus_s))
    # Denominator: asymptotic susceptibility
    chi_inf = np.diag(Gamma_inv)
    # Element-wise normalization
    intRts_norm1 = chi / chi_inf
    return intRts_norm1

def intRts_norm2_Langevin_ND(Gamma, sigma, V_0, t, s):
    """
    Normalizes susceptibility using noise amplitude and autocorrelation:
        tilde_chi_i = (sigma_i^2 * chi_i(t,s)) / (2 * C_i(t,t))
    """
    Ctt_diag = Cts_Langevin_ND(Gamma, sigma, V_0, t, t)
    exp_Gamma_t_minus_s = exp_scaling_squaring(-Gamma * (t - s))
    sigma_squared = sigma**2

    I = np.eye(Gamma.shape[0])
    Gamma_inv = np.linalg.inv(Gamma)
    
    chi = np.diag(Gamma_inv @ (I - exp_Gamma_t_minus_s))  # chi_i(t,s)
    
    # Noise-normalized susceptibility
    intRts_norm2 = (sigma_squared * chi) / (2 * Ctt_diag)
    return intRts_norm2

### Compute A ###############################################
@jit(forceobj=True)
def Ats_Langevin_ND(Gamma, sigma, V_0, t, s):
    term1 = dsCts_Langevin_ND(Gamma, sigma, V_0, t, s)
    term2 = -dtCts_Langevin_ND(Gamma, sigma, V_0, t, s)
    term3 = -sigma**2 * Rts_Langevin_ND(Gamma, t, s)
    Vts = term1 + term2 + term3
    return Vts

@jit(forceobj=True)
def Ats_Langevin_ND_numerical(Gamma, sigma, V_0, t, s):
    term1 = dsCts_Langevin_ND_numerical(Gamma, sigma, V_0, t, s)
    term2 = -dtCts_Langevin_ND_numerical(Gamma, sigma, V_0, t, s)
    term3 = -sigma**2 * Rts_Langevin_ND(Gamma, t, s)
    Vts = term1 + term2 + term3
    return Vts

### Compute X ###############################################
@jit(forceobj=True)
def Xts_Langevin_ND(Gamma, sigma, V_0, t, s):
    dsCts = dsCts_Langevin_ND(Gamma, sigma, V_0, t, s)
    Rts = Rts_Langevin_ND(Gamma, t, s)
    Xts = (sigma**2 / 2) * Rts / dsCts
    return Xts

@jit(forceobj=True)
def Xts_Langevin_ND_numerical(Gamma, sigma, V_0, t, s):
    dsCts = dsCts_Langevin_ND_numerical(Gamma, sigma, V_0, t, s)
    Rts = Rts_Langevin_ND(Gamma, t, s)
    Xts = (sigma**2 / 2) * Rts / dsCts
    return Xts

### Compute V ###############################################
@jit(forceobj=True)
def Vts_Langevin_ND(Gamma, sigma, V_0, t, s):
    term1 = dsCts_Langevin_ND(Gamma, sigma, V_0, t, s)
    term2 = sigma**2/2 * Rts_Langevin_ND(Gamma, t, s)
    Vts = term1 + term2
    return Vts

@jit(forceobj=True)
def Vts_Langevin_ND_numerical(Gamma, sigma, V_0, t, s):
    term1 = dsCts_Langevin_ND_numerical(Gamma, sigma, V_0, t, s)
    term2 = sigma**2/2 * Rts_Langevin_ND(Gamma, t, s)
    Vts = term1 + term2
    return Vts

### Compute I ###############################################
@jit(forceobj=True)
def Its_Langevin_ND(Gamma, sigma, V_0, t, s):
    # Compute C_i(t, t) and C_i(t, s)
    C_tt = Cts_Langevin_ND(Gamma, sigma, V_0, t, t)
    C_ts = Cts_Langevin_ND(Gamma, sigma, V_0, t, s)
    # Compute the identity matrix
    I = np.eye(Gamma.shape[0])
    # Compute the matrix exponential
    exp_Gamma_t_minus_s = exp_scaling_squaring(-Gamma * (t - s))
    # Compute the term for the subtraction
    term3 = np.linalg.inv(Gamma) @ (I - exp_Gamma_t_minus_s)
    term3_ii = np.diag(term3)  # Extract diagonal elements
    # Calculate I_i(t, s)
    Its = C_tt - C_ts - (sigma**2 / 2) * term3_ii
    return Its

### Compute Inorm1 ###############################################
### Inorm1(t,s) = I(t,s)/(sigma^2 / 2 * int_R(t,s))
@jit(forceobj=True)
def Its_norm1_Langevin_ND(Gamma, sigma, V_0, t, s, eps=0):
    # Compute C_i(t, t) and C_i(t, s)
    C_tt = Cts_Langevin_ND(Gamma, sigma, V_0, t, t)
    C_ts = Cts_Langevin_ND(Gamma, sigma, V_0, t, s)
    # Compute the identity matrix
    I = np.eye(Gamma.shape[0])
    # Compute the matrix exponential
    exp_Gamma_t_minus_s = exp_scaling_squaring(-Gamma * (t - s))
    # Compute the term for the subtraction
    term3 = np.linalg.inv(Gamma) @ (I - exp_Gamma_t_minus_s)
    term3_ii = np.diag(term3)  # Extract diagonal elements
    # Compute int_R
    intRts = np.diag(np.linalg.inv(Gamma) @ (I - exp_Gamma_t_minus_s))
    # Calculate I_i(t, s)
    Inorm1 = (C_tt - C_ts - (sigma**2 / 2) * term3_ii) / (sigma**2 / 2 * intRts + eps)
    return Inorm1

### Compute Inorm2 ###############################################
### Inorm2(t,s) = I(t,s)/(C(t,t) + eps)
@jit(forceobj=True)
def Its_norm2_Langevin_ND(Gamma, sigma, V_0, t, s, eps=0):
    # Compute C_i(t, t) and C_i(t, s)
    C_tt = Cts_Langevin_ND(Gamma, sigma, V_0, t, t)
    C_ts = Cts_Langevin_ND(Gamma, sigma, V_0, t, s)
    # Compute the identity matrix
    I = np.eye(Gamma.shape[0])
    # Compute the matrix exponential
    exp_Gamma_t_minus_s = exp_scaling_squaring(-Gamma * (t - s))
    # Compute the term for the subtraction
    term3 = np.linalg.inv(Gamma) @ (I - exp_Gamma_t_minus_s)
    term3_ii = np.diag(term3)  # Extract diagonal elements
    # Calculate I_i(t, s)
    Inorm2 = (C_tt - C_ts - (sigma**2 / 2) * term3_ii) / (C_tt + eps)
    return Inorm2

### Compute Inorm3 ###############################################
### Inorm3(t,s) = I(t,s)/(C(t,t) - C(t,s) + eps)
@jit(forceobj=True)
def Its_norm3_Langevin_ND(Gamma, sigma, V_0, t, s, eps=0):
    # Compute C_i(t, t) and C_i(t, s)
    C_tt = Cts_Langevin_ND(Gamma, sigma, V_0, t, t)
    C_ts = Cts_Langevin_ND(Gamma, sigma, V_0, t, s)
    # Compute the identity matrix
    I = np.eye(Gamma.shape[0])
    # Compute the matrix exponential
    exp_Gamma_t_minus_s = exp_scaling_squaring(-Gamma * (t - s))
    # Compute the term for the subtraction
    term3 = np.linalg.inv(Gamma) @ (I - exp_Gamma_t_minus_s)
    term3_ii = np.diag(term3)  # Extract diagonal elements
    # Calculate I_i(t, s)
    Inorm2 = (C_tt - C_ts - (sigma**2 / 2) * term3_ii) / (C_tt - C_ts + eps)
    return Inorm2

###############################################################
### Numerical integration of V(t,s) ###########################
from scipy.integrate import quad
def compute_I_as_numerical_integration_of_V(t, s, Gamma, V_0, sigma):
    N = Gamma.shape[0]
    I_numerical = np.zeros(N)
    for i in range(N):
        def V_func(s_prime):
            return Vts_Langevin_ND(Gamma, sigma, V_0, t, s_prime)[i]
        I_numerical[i], _ = quad(V_func, s, t)  # Numerically integrate V_i(t, s')
    return I_numerical

def numerical_integration_of_Vts_in_s(t, Gamma, V_0, sigma):
    N = Gamma.shape[0]
    I_numerical = np.zeros(N)
    for i in range(N):
        def V_func(s_prime):
            return Vts_Langevin_ND(Gamma, sigma, V_0, t, s_prime)[i]
        I_numerical[i], _ = quad(V_func, 0, t)  # Numerically integrate V_i(t, s')
    return I_numerical

def numerical_integration_of_Vts_in_s_and_in_t(tmax, Gamma, V_0, sigma):
    N = Gamma.shape[0]
    I_numerical = np.zeros(N)
    for i in range(N):
        def V_func(t_prime):
            return numerical_integration_of_Vts_in_s(t_prime, Gamma, V_0, sigma)[i]
        I_numerical[i], _ = quad(V_func, 0, tmax)
    return I_numerical

###############################################################
### exp(Gamma) approximation ##################################

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

###############################################################
### Downsampling ##############################################

# @jit(nopython=True)
@njit(parallel=True)
def FDT_Langevin_ND_downsample(nsim, Gamma, sigma, v0std, v0bias, Thermalization, duration, integstep, Mds=20):
    Ndim = np.shape(Gamma)[0]
    nsteps = int(duration / integstep) + 1

    # Downsample
    N = nsteps
    M = closest_valid_M(N, Mds)
    nsteps_down = (N - 1) // M + 1
    nsteps = nsteps_down

    Cts = np.zeros((Ndim, nsteps, nsteps))
    Rts = np.zeros((Ndim, nsteps, nsteps))
    Ats = np.zeros((Ndim, nsteps, nsteps))
    V_0 = np.zeros_like(Gamma)
    Gamma = np.ascontiguousarray(Gamma)  # Make Gamma contiguous

    inv_nsim = 1.0 / nsim
    inv_nsim_sigma2_sqrtdt = np.ones(Ndim) / (nsim * sigma**2 * np.sqrt(integstep))

    for i in range(nsim):
        # Set initial conditions
        initcond = v0std * np.random.standard_normal(Ndim) + v0bias
        if Thermalization == 1:
            # Discard first "tfinal/dt" time-points
            tfinal = 100
            dt = integstep
            v0 = initcond
            vsim, _ = Integrate_Langevin_ND_Optimized(Gamma, sigma, initcond=v0, duration=tfinal, integstep=dt)
            initcond = vsim[:, -1]
        V_0 += np.outer(initcond,initcond)

        # Integrate Langevin dynamics
        Xsim, noise = Integrate_Langevin_ND_Optimized(Gamma, sigma, initcond, duration, integstep)
        Xsim = np.ascontiguousarray(Xsim[:, ::M])  # Downsample and make contiguous
        noise = np.ascontiguousarray(noise[:, ::M])  # Downsample and make contiguous

        for i in prange(Ndim):
            for t in range(nsteps):
                for s in range(nsteps):
                    Xsim_it = Xsim[i, t]
                    Xsim_is = Xsim[i, s]
                    
                    Cts[i, t, s] += Xsim_it * Xsim_is
                    Rts[i, t, s] += Xsim_it * noise[i, s]
                    
                    for j in range(Ndim):
                        Gamma_ij = Gamma[i, j]
                        Xsim_jt = Xsim[j, t]
                        Xsim_js = Xsim[j, s]
                        
                        Ats[i, t, s] += Gamma_ij * (Xsim_jt * Xsim_is - Xsim_js * Xsim_it)
                    
    # # Ensemble average
    Cts *= inv_nsim
    for i in prange(Ndim):
        Rts[i,:,:] *= inv_nsim_sigma2_sqrtdt[i]
    Ats *= inv_nsim
    V_0 *= inv_nsim
    return Cts, Rts, Ats, V_0, M


# @jit(nopython=True)
@njit(parallel=True)
def FDT_Hopf_ND_downsample(nsim, BIFURCATION_PARAM_a, NODES_FREQS_w, CONNECTIVITY_C, GLOBAL_COUPLING_G, NOISE_SIGMA, v0std, v0bias, Thermalization, duration, integstep, Mds=20):
    Ndim = np.shape(CONNECTIVITY_C)[0]
    nsteps = int(duration / integstep) + 1

    # Downsample
    N = nsteps
    M = closest_valid_M(N, Mds)
    nsteps_down = (N - 1) // M + 1
    nsteps = nsteps_down

    Cts = np.zeros((Ndim, nsteps, nsteps))
    Rts = np.zeros((Ndim, nsteps, nsteps))
    # Ats = np.zeros((Ndim, nsteps, nsteps))
    V_0 = np.zeros_like(CONNECTIVITY_C)
    CONNECTIVITY_C = np.ascontiguousarray(CONNECTIVITY_C)  # Make Connectivity contiguous

    inv_nsim = 1.0 / nsim
    inv_nsim_sigma2_sqrtdt = np.ones(Ndim) / (nsim * NOISE_SIGMA**2 * np.sqrt(integstep))

    # For the other Hopf
    wC = CONNECTIVITY_C
    row_sums = np.sum(wC, axis=1).reshape(-1, 1)
    sumC = np.column_stack((row_sums, row_sums))
    omega_xy = np.column_stack((NODES_FREQS_w, NODES_FREQS_w))

    for i in range(nsim):
        # Set initial conditions
        X0 = v0std * np.random.standard_normal(Ndim) + v0bias
        Y0 = v0std * np.random.standard_normal(Ndim) + v0bias
        if Thermalization == 1:
            # Discard first "tfinal/dt" time-points
            term_time = 100
            dt = integstep
            Xsim, Ysim, _ = Integrate_Hopf_ND_Optimized(BIFURCATION_PARAM_a, NODES_FREQS_w, CONNECTIVITY_C, GLOBAL_COUPLING_G, NOISE_SIGMA, X0, Y0, term_time, dt)

            # z0 = np.column_stack((X0, Y0))
            # Xsim, Ysim, _, _ = Integrate_Hopf_ND_Optimized_2(ahopf=BIFURCATION_PARAM_a, omega=omega_xy, sigma=NOISE_SIGMA, dt=dt, Tmax=term_time, wC=wC, sumC=sumC, z0=z0)

            X0 = Xsim[:, -1]
            Y0 = Ysim[:, -1]
        V_0 += np.outer(X0,X0)

        # Integrate Hopf model
        Xsim, _, noise = Integrate_Hopf_ND_Optimized(BIFURCATION_PARAM_a, NODES_FREQS_w, CONNECTIVITY_C, GLOBAL_COUPLING_G, NOISE_SIGMA, X0, Y0, duration, integstep)

        # z0 = np.column_stack((X0, Y0))
        # Xsim, _, _, noise = Integrate_Hopf_ND_Optimized_2(ahopf=BIFURCATION_PARAM_a, omega=omega_xy, sigma=NOISE_SIGMA, dt=integstep, Tmax=duration, wC=wC, sumC=sumC, z0=z0)

        Xsim = np.ascontiguousarray(Xsim[:, ::M])  # Downsample and make contiguous
        noise = np.ascontiguousarray(noise[:, ::M])  # Downsample and make contiguous

        for i in prange(Ndim):
            for t in range(nsteps):
                for s in range(0,t):
                    Xsim_it = Xsim[i, t]
                    Xsim_is = Xsim[i, s]
                    
                    Cts[i, t, s] += Xsim_it * Xsim_is
                    Rts[i, t, s] += Xsim_it * noise[i, s]
                    
                    # for j in range(Ndim):
                    #     Gamma_ij = Gamma[i, j]
                    #     Xsim_jt = Xsim[j, t]
                    #     Xsim_js = Xsim[j, s]
                    #     Ats[i, t, s] += (Gamma_ij * Xsim_jt * Xsim_is - Gamma_ij * Xsim_js * Xsim_it)
                    
    # # Ensemble average
    Cts *= inv_nsim
    for i in prange(Ndim):
        Rts[i,:,:] *= inv_nsim_sigma2_sqrtdt[i]
    V_0 *= inv_nsim
    # Ats *= inv_nsim
    return Cts, Rts, V_0, M #, Ats

@jit(nopython=True)
def closest_valid_M(N, M):
    if (N-1) % M == 0:
        return M
    # Find the closest valid M less than or equal to the given M
    for lower_M in range(M, 0, -1):
        if (N-1) % lower_M == 0:
            closest_lower_M = lower_M
            break
    # Find the closest valid M greater than the given M
    for upper_M in range(M, N):
        if (N-1) % upper_M == 0:
            closest_upper_M = upper_M
            break
    # Compare which valid M is closer to the given M
    if abs(M - closest_lower_M) <= abs(M - closest_upper_M):
        return closest_lower_M
    else:
        return closest_upper_M


###############################################################
### Other functions ###########################################

def plot_C_matrices(Cts_sim, max_columns=2):
    Ndim = Cts_sim.shape[0]
    nsteps = Cts_sim.shape[1]
    # Calculate the number of rows needed
    nrows = (Ndim + max_columns - 1) // max_columns
    fig, axes = plt.subplots(nrows, max_columns, figsize=(10, nrows * 4))
    fig.tight_layout(pad=5.0)
    # Flatten the axes array for easier indexing
    axes = axes.flatten()
    for i in range(Ndim):
        ax = axes[i]
        ax.set_title(r'$C(t,s)[node {}]$'.format(i+1))
        ax.set_frame_on(False)
        ax.set_xlabel(r'$s_n$')
        ax.set_ylabel(r'$t_n$')
        mask = np.transpose(np.tri(nsteps, k=-1))
        Ctsmask = np.ma.array(Cts_sim[i, :, :], mask=mask)
        im = ax.imshow(Ctsmask, interpolation='None', aspect='auto', cmap='viridis')
        # Add a colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')
    # Hide any unused subplots
    for j in range(Ndim, len(axes)):
        fig.delaxes(axes[j])
    plt.show()

def plot_matrices(Cts_sim, max_columns=2, title='C'):
    Ndim = Cts_sim.shape[0]
    nsteps = Cts_sim.shape[1]
    # Calculate the number of rows needed
    nrows = (Ndim + max_columns - 1) // max_columns
    fig, axes = plt.subplots(nrows, max_columns, figsize=(10, nrows * 4))
    fig.tight_layout(pad=5.0)
    # Flatten the axes array for easier indexing
    axes = axes.flatten()
    for i in range(Ndim):
        ax = axes[i]
        ax.set_title(r'${}[node {}]$'.format(title, i+1))
        ax.set_frame_on(False)
        ax.set_xlabel(r'$s_n$')
        ax.set_ylabel(r'$t_n$')
        mask = np.transpose(np.tri(nsteps, k=-1))
        Ctsmask = np.ma.array(Cts_sim[i, :, :], mask=mask)
        im = ax.imshow(Ctsmask, interpolation='None', aspect='auto', cmap='viridis')
        # Add a colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')
    # Hide any unused subplots
    for j in range(Ndim, len(axes)):
        fig.delaxes(axes[j])
    plt.show()

def plot_C_matrices_box(Cts_sim, max_columns=2, rect_params=None):
    """
    Plot matrices with an optional rectangle overlay.

    Parameters:
    - Cts_sim: 3D numpy array, shape (Ndim, nsteps, nsteps)
    - max_columns: int, maximum number of columns in the subplot grid
    - rect_params: tuple, containing (x, y, width, height) for the rectangle
                   position and size in data coordinates.
    """
    Ndim = Cts_sim.shape[0]
    nsteps = Cts_sim.shape[1]
    nrows = (Ndim + max_columns - 1) // max_columns
    fig, axes = plt.subplots(nrows, max_columns, figsize=(10, nrows * 4))
    fig.tight_layout(pad=5.0)
    axes = axes.flatten()
    
    for i in range(Ndim):
        ax = axes[i]
        ax.set_title(r'$C(t,s)[node {}]$'.format(i+1))
        ax.set_frame_on(False)
        ax.set_xlabel(r'$s_n$')
        ax.set_ylabel(r'$t_n$')
        mask = np.transpose(np.tri(nsteps, k=0))
        Ctsmask = np.ma.array(Cts_sim[i, :, :], mask=mask)
        im = ax.imshow(Ctsmask, interpolation='None', aspect='auto', cmap='viridis')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')

        if rect_params:
            x, y, width, height = rect_params
            rect = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
    
    for j in range(Ndim, len(axes)):
        fig.delaxes(axes[j])
    
    plt.show()
