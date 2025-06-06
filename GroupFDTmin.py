# This is the main script for the 'Group' Analytical Hopf FDT
# I will try to keep it as minimal as possible
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import numpy as np
import os
import scipy.io
import neuronumba
import scipy.integrate as integrate
from scipy.linalg import expm
from scipy.linalg import solve_continuous_lyapunov
from DataLoaders.baseDataLoader import DataLoader
import ADNI_A
from functions_FDT_numba_v8 import *
from functions_boxplots_WN3_v0 import *
from functions_violinplots_WN3_v0 import *
import filterps
from functions_violinplots_WN3_v0 import plot_violins_HC_MCI_AD
from typing import Union
import p_values as p_values
import sys
import statannotations_permutation as perm

np.random.seed(42)

script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
os.chdir(script_dir)
print("Working directory changed to:", os.getcwd())

def calc_H_freq(
        all_HC_fMRI: Union[np.ndarray, dict], 
        tr: float, 
        version: filterps.FiltPowSpetraVersion=filterps.FiltPowSpetraVersion.v2021
    ):
        """
        Compute H freq for each node. 
        
        Parameters
        ----------
        all_HC_fMRI: The fMRI of the "health control" group. Can be given in a dictionaray format, 
                     or in an array format (subject, time, node).
                     NOTE: that the signals must already be filitered. 
        tr: TR in milliseconds
        version: Version of FiltPowSpectra to use

        Returns
        -------
        The h frequencies for each node
        """
        f_diff = filterps.filt_pow_spetra_multiple_subjects(all_HC_fMRI, tr, version)
        return 2 * np.pi * f_diff  # omega

NPARCELLS = 379

groups = ['HC', 'MCI', 'AD']
DL = ADNI_A.ADNI_A()

# Loading the timeseries data for all subjects
HC_IDs = DL.get_groupSubjects('HC')
HC_MRI = {}
for subject in HC_IDs:
    data = DL.get_subjectData(subject)
    HC_MRI[subject] = data[subject]['timeseries'].transpose()

MCI_IDs = DL.get_groupSubjects('MCI')
MCI_MRI = {}
for subject in MCI_IDs:
    data = DL.get_subjectData(subject)
    MCI_MRI[subject] = data[subject]['timeseries'].transpose()

AD_IDs = DL.get_groupSubjects('AD')
AD_MRI = {}
for subject in AD_IDs:
    data = DL.get_subjectData(subject)
    AD_MRI[subject] = data[subject]['timeseries'].transpose()

# Here we load the Effective Connectivity data
EC_HC_data = scipy.io.loadmat('ADNI-A_DATA/EC_filterted/HC_FDT_results_filters0109.mat')
EC_MCI_data = scipy.io.loadmat('ADNI-A_DATA/EC_filterted/MCI_FDT_results_filters0109.mat')
EC_AD_data = scipy.io.loadmat('ADNI-A_DATA/EC_filterted/AD_FDT_results_filters0109.mat')

Ceffgroup_HC = EC_HC_data['Ceff_subjects']
omega_HC = calc_H_freq(HC_MRI, 3000, filterps.FiltPowSpetraVersion.v2021)
Ceffgroup_MCI = EC_MCI_data['Ceff_subjects']
omega_MCI = calc_H_freq(MCI_MRI, 3000, filterps.FiltPowSpetraVersion.v2021)
Ceffgroup_AD = EC_AD_data['Ceff_subjects']
omega_AD = calc_H_freq(AD_MRI, 3000, filterps.FiltPowSpetraVersion.v2021)


#Hopf model parameters
aparam = -0.02
gconst = 1.0
TR = 2
avec = aparam * np.ones(NPARCELLS) # Possibility for different a values
Ndim = 2 * NPARCELLS
sigma = 0.08
sigma = sigma * np.ones(NPARCELLS)
sigma = np.append(sigma, sigma)
v0std = sigma
v0bias = 0.0
t0 = 0
tfinal = 200
dt = 1.0
times = np.arange(t0, tfinal+dt, dt)
nsteps = len(times)
t_times = times[0:nsteps]
s_times = times[0:nsteps]
tmax = 100
ts = 0

for i in groups:
    
    if i == 'HC':
        omega = omega_HC
        Ceff = np.mean(Ceffgroup_HC, axis=0)
    elif i == 'MCI':
        omega = omega_HC
        Ceff = np.mean(Ceffgroup_MCI, axis=0)
    elif i == 'AD':
        omega = omega_HC
        Ceff = np.mean(Ceffgroup_AD, axis=0)

    Gamma = -construct_matrix_A(avec, omega, Ceff, gconst)

    eigenvalues_of_Gamma = np.linalg.eigvals(Gamma)
    tol = 1e-12
    is_stable = np.all(np.real(eigenvalues_of_Gamma) > tol)
    print(is_stable)

    v0 = v0std * np.random.standard_normal(Ndim) + v0bias
    vsim, noise = Integrate_Langevin_ND_Optimized(Gamma, sigma, initcond=v0, duration=tfinal, integstep=dt)
    
    #ndim = np.shape(Gamma)[0]
    V_0 = V_0_calculation_v0gauss(v0std, v0bias, Ndim)


    Its_calc = np.zeros((NPARCELLS,len(t_times)))
    I_tmax_s0 = Its_Langevin_ND(Gamma, sigma, V_0, tmax, ts)[0:NPARCELLS]
    if i == 'HC':
        I_HC = I_tmax_s0
    elif i == 'MCI':   
        I_MCI = I_tmax_s0
    elif i == 'AD':
        I_AD = I_tmax_s0

    for t in range(len(t_times)):
        Its_calc[:,t] = Its_Langevin_ND(Gamma, sigma, V_0, t_times[t], s_times[0])[0:NPARCELLS] # I keep ONLY for the x-components

    for par in range(NPARCELLS):
        plt.plot(times, Its_calc[par,:])
    plt.gca().set_prop_cycle(None)
    for par in range(NPARCELLS):
        plt.plot(times[-1],Its_calc[par,-1],'o')

    plt.title(r'FDT: $I(t,s=0)$')
    plt.xlabel(r'$t$ (s)')
    plt.ylabel(r'FDT: $I$')
    plt.show()

    plt.plot(np.arange(NPARCELLS)+1, I_tmax_s0,'.-',color='gray',alpha=0.7)
    plt.gca().set_prop_cycle(None)
    for par in range(NPARCELLS):
        plt.plot(par+1,I_tmax_s0[par],'o-')

    plt.title(r'FDT: $M_I=I(t=t_{max},s=0)$')
    plt.xlabel('Node')
    plt.ylabel(r'FDT: $M_I$')
    x_ticks = 10 if NPARCELLS > 20 else 1
    plt.xticks(np.arange(0, NPARCELLS+1, x_ticks))
    plt.show()

data = pd.DataFrame({
        "value": np.concatenate([I_HC, I_MCI, I_AD]),
        "cond": ["HC"] * len(I_HC) + ["MCI"] * len(I_MCI) + ["AD"] * len(I_AD),
    })
print(data.shape)
print("IHC",I_HC.shape)

# Create dictionary like in loadResultsCohort demo
resI = {
    'HC': I_HC,
    'MCI': I_MCI,
    'AD': I_AD
}

# Plot
plt.rcParams.update({'font.size': 15})
p_values.plotComparisonAcrossLabels2(
    resI,
    custom_test=perm.custom_permutation(),
    columnLables=['HC', 'MCI', 'AD'],
    graphLabel='FDT I(tmax, 0) Parcels'
)

print('done!!')