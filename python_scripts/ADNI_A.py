# =====================================================================================
# Methods to input AD data
# Subjects: HC 17, MCI 9, AD 10 - RoIs: 379 - TR = 3 - timepoints: 197 (but 2 have 950-ish)
# Info for each subject: timeseries, ABeta, Tau, SC
#
# The ADNI-1 dataset was graciously provided by Leon Stefanovsi and Petra Ritter
#
# =====================================================================================
import numpy as np
import os, csv
from pathlib import Path
import sys
# Ensure repo root is on sys.path for module imports
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, repo_root)

# Import central paths from config
from support_files.config import (
    connectome_dir,
    raw_data_dir,
    produced_data_dir,
)

from DataLoaders.baseDataLoader import DataLoader
import DataLoaders.Parcellations.Glasser379 as Glasser379

# Make sure data folders exist
Path(raw_data_dir).mkdir(parents=True, exist_ok=True)
Path(produced_data_dir).mkdir(parents=True, exist_ok=True)

# Set base folder for data processing
base_folder = connectome_dir  # or use raw_data_dir if that makes more sense contextually

# Print for debug (optional)
if __name__ == '__main__':
    print("Using base folder:", base_folder)
# ==========================================================================
# ==========================================================================
# ==========================================================================


def characterizeConnectivityMatrix(C):
    return np.max(C), np.min(C), np.average(C), np.std(C), np.max(np.sum(C, axis=0)), np.average(np.sum(C, axis=0))


def checkClassifications(subjects,printInfo=True):
    # ============================================================================
    # This code is to check whether we have the information of the type of subject
    # They can be one of:
    # Healthy Controls (HC), Mild Cognitive Impairment (MCI), Alzheimer Disease (AD) or Significant Memory Concern (SMC)
    # ============================================================================
    input_classification = csv.reader(open(os.path.join(os.path.dirname(base_folder), "subjects.csv"), 'r'))

    classification = dict((rows[0],rows[1]) for rows in input_classification)
    mistery = []
    for pos, subject in enumerate(subjects):
        if subject in classification:
            if printInfo: print('{}: Subject {} classified as {}'.format(pos, subject, classification[subject]))
        else:
            if printInfo: print('{}: Subject {} NOT classified'.format(pos, subject))
            mistery.append(subject)
    if printInfo: print("Misisng {} subjects:".format(len(mistery)), mistery)
    if printInfo: print()
    return classification


def getClassifications():
    # ============================================================================
    # This code is to check whether we have the information of the type of subject
    # They can be one of:
    # Healthy Controls (HC), Mild Cognitive Impairment (MCI), Alzheimer Disease (AD) or Significant Memory Concern (SMC)
    # ============================================================================
    classification = {}
    input_classification = csv.reader(open(os.path.join(os.path.dirname(base_folder), "subjects.csv"), 'r'))

    #input_classification = csv.reader(open(base_folder+"subjects.csv", 'r'))
    for row in input_classification:
        classification[row[0]] = row[1]
    return classification

# =====================================================================================
# Methods to input AD data
# =====================================================================================
def loadBurden(subject, modality, baseFolder, normalize=True):
    pet_path = os.path.dirname(baseFolder) + "/PET_loads/"+subject+"/PET_PVC_MG/" + modality
    RH_pet = np.loadtxt(pet_path+"/"+"L."+modality+"_load_MSMAll.pscalar.txt")
    LH_pet = np.loadtxt(pet_path+"/"+"R."+modality+"_load_MSMAll.pscalar.txt")
    subcort_pet = np.loadtxt(pet_path+"/"+modality+"_load.subcortical.txt")[-19:]
    all_pet = np.concatenate((LH_pet,RH_pet,subcort_pet))
    if normalize:
        normalizedPet = all_pet / np.max(all_pet)  # We need to normalize the individual burdens for the further optimization steps...
    else:
        normalizedPet = all_pet
    return normalizedPet


# ===================== compute the Avg SC matrix over the HC sbjects
def computeAvgSC_HC_Matrix(classification, baseFolder):
    HC = [subject for subject in classification.keys() if classification[subject] == 'HC']
    print("SC + HC: {} (0)".format(HC[0]))
    sc_folder = os.path.join(base_folder, HC[0], "DWI_processing")

    #sc_folder = baseFolder+'/'+HC[0]+"/DWI_processing"
    SC = np.loadtxt(sc_folder+"/connectome_weights.csv")

    sumMatrix = SC
    for subject in HC[1:]:
        print("SC + HC: {}".format(subject))
        sc_folder = os.path.join(base_folder, HC[0], "DWI_processing")
        #sc_folder = baseFolder+'/'+subject+"/DWI_processing"
        SC = np.loadtxt(sc_folder+"/connectome_weights.csv")
        sumMatrix += SC
    return sumMatrix / len(HC)  # but we normalize it afterwards, so we probably do not need this...


# ===================== Load one specific subject data
def loadSubjectData(subject, correcSCMatrix=True, normalizeBurden=True):
    sc_folder = os.path.join(base_folder, subject, "DWI_processing")
    #sc_folder = base_folder + 'connectomes/'+subject+"/DWI_processing/"
    SC = np.loadtxt(sc_folder + "/connectome_weights.csv")
    if correcSCMatrix:
        SCnorm = correctSC(SC)
    else:
        SCnorm = np.log(SC + 1)

    abeta_burden = loadBurden(subject, "Amyloid", base_folder, normalize=normalizeBurden)
    tau_burden = loadBurden(subject, "Tau", base_folder, normalize=normalizeBurden)

    fMRI_path = os.path.dirname(base_folder)+"/fMRI/"+subject+"/MNINonLinear/Results/Restingstate/"
    series = np.loadtxt(fMRI_path+subject+"_Restingstate_Atlas_MSMAll_hp2000_clean.ptseries.txt")
    subcSeries = np.loadtxt(fMRI_path+subject+"_Restingstate_Atlas_MSMAll_hp2000_clean_subcort.ptseries.txt")
    fullSeries = np.concatenate((series,subcSeries))

    return SCnorm, abeta_burden, tau_burden, fullSeries


# ===================== Load all fMRI data
def load_fullCohort_fMRI(classification, baseFolder, cohort='HC'):
    cohortSet = [subject for subject in classification.keys() if classification[subject] == cohort]
    all_fMRI = {}
    for subject in cohortSet:
        print(f"fMRI {cohort}: {subject}")
        fMRI_path = baseFolder + "/fMRI/" + subject + "/MNINonLinear/Results/Restingstate/"
        series = np.loadtxt(fMRI_path+subject+"_Restingstate_Atlas_MSMAll_hp2000_clean.ptseries.txt")
        subcSeries = np.loadtxt(fMRI_path+subject+"_Restingstate_Atlas_MSMAll_hp2000_clean_subcort.ptseries.txt")
        fullSeries = np.concatenate((series, subcSeries))
        all_fMRI[subject] = fullSeries
    return all_fMRI


# ===================== Load all fMRI data
def load_all_HC_fMRI(classification, baseFolder):
    load_fullCohort_fMRI(classification, baseFolder, cohort='HC')


# ===================== Normalize a SC matrix
normalizationFactor = 0.2
avgHuman66 = 0.0035127188987848714
areasHuman66 = 66  # yeah, a bit redundant... ;-)
maxNodeInput66 = 0.7275543904602363
def correctSC(SC):
    N = SC.shape[0]
    logMatrix = np.log(SC+1)
    # areasSC = logMatrix.shape[0]
    # avgSC = np.average(logMatrix)
    # === Normalization ===
    # finalMatrix = normalizationFactor * logMatrix / logMatrix.max()  # normalize to the maximum, as in Gus' codes
    # finalMatrix = logMatrix * avgHuman66/avgSC * (areasHuman66*areasHuman66)/(areasSC * areasSC)  # normalize to the avg AND the number of connections...
    maxNodeInput = np.max(np.sum(logMatrix, axis=0))  # This is the same as np.max(logMatrix @ np.ones(N))
    finalMatrix = logMatrix * maxNodeInput66 / maxNodeInput
    return finalMatrix


def analyzeMatrix(name, C):
    max, min, avg, std, maxNodeInput, avgNodeInput = characterizeConnectivityMatrix(C)
    print(name + " => Shape:{}, Max:{}, Min:{}, Avg:{}, Std:{}".format(C.shape, max, min, avg, std), end='')
    print("  => impact=Avg*#:{}".format(avg*C.shape[0]), end='')
    print("  => maxNodeInputs:{}".format(maxNodeInput), end='')
    print("  => avgNodeInputs:{}".format(avgNodeInput))


# This is used to avoid (almost) "infinite" computations for some cases (i.e., subjects) that have fMRI
# data that is way longer than any other subject, causing almost impossible computations to perform,
# because they last several weeks (~4 to 6), which seems impossible to complete with modern Windows SO,
# which restarts the computer whenever it wants to perform supposedly "urgent" updates...
force_Tmax = True


# This method is to perform the timeSeries cutting when excessively long...
def cutTimeSeriesIfNeeded(timeseries, limit_forcedTmax, printInfo=True):
    if force_Tmax and timeseries.shape[1] > limit_forcedTmax:
        if printInfo: print(f"cutting lengthy timeseries: {timeseries.shape[1]} to {limit_forcedTmax}")
        timeseries = timeseries[:,0:limit_forcedTmax]
    return timeseries


# --------------------------------------------------
# Classify subject information into {HC, MCI, AD}
# --------------------------------------------------
def getCohortSubjects(cohort):
    return [s for s in classification if classification[s] == cohort]


subjects = [os.path.basename(f.path) for f in os.scandir(base_folder) if f.is_dir()]
classification = checkClassifications(subjects,printInfo=False)
HCSubjects = [s for s in classification if classification[s] == 'HC']
ADSubjects = [s for s in classification if classification[s] == 'AD']
MCISubjects = [s for s in classification if classification[s] == 'MCI']
#print(f"We have {len(HCSubjects)} HC, {len(MCISubjects)} MCI and {len(ADSubjects)} AD \n")
# print("HCSubjects:", HCSubjects)
# print("ADSubjects", ADSubjects)
# print("MCISubjects", MCISubjects)
# allStudySubjects = HCSubjects + MCISubjects + ADSubjects

dataSetLabels = ['HC', 'MCI', 'AD']


# ================================================================================================================
# ================================================================================================================
# Loading generalization layer:
# These methods are used for the sole purpose of homogenizing data loading across projects
# ================================================================================================================
# ================================================================================================================
class ADNI_A(DataLoader):
    def __init__(self, use360=False, correcSCMatrix=True, normalizeBurden=True, cutTimeSeries=True):
        self.use360 = use360
        self.correcSCMatrix = correcSCMatrix
        self.normalizeBurden = normalizeBurden
        global force_Tmax
        force_Tmax = cutTimeSeries
        self.BOLD_length = 197

    def name(self):
        return 'ADNI_A'

    def set_basePath(self, path):
        global WholeBrainFolder, base_folder
        WholeBrainFolder = path
        base_folder = WholeBrainFolder + "Data_Raw/from_Ritter/"

    def TR(self):
        return 3.

    def N(self):
        if self.use360:
            return 360
        return 379 # 360 cortical + 19 subcortical regions

    # get_fullGroup_data: convenience method to load all data for a given subject group
    def get_fullGroup_data(self, group):
        groupFMRI = load_fullCohort_fMRI(classification, base_folder, cohort=group)
        for s in groupFMRI:
            groupFMRI[s] = {'timeseries': cutTimeSeriesIfNeeded(groupFMRI[s], self.BOLD_length)}
        return groupFMRI

    def get_AvgSC_ctrl(self, normalized=False):
        avgMatrix = computeAvgSC_HC_Matrix(classification, base_folder+"connectomes/")
        if normalized:
            return correctSC(avgMatrix)
        else:
            return avgMatrix

    def get_groupSubjects(self, group):
        return getCohortSubjects(group)

    def get_groupLabels(self):
        return dataSetLabels

    def get_classification(self):
        return classification

    def get_subjectData(self, subjectID, printInfo=True):
        # 1st, load
        SCnorm, abeta_burden, tau_burden, timeseries = loadSubjectData(subjectID,
                                                                       correcSCMatrix=self.correcSCMatrix,
                                                                       normalizeBurden=self.normalizeBurden)
        # 2nd, cut
        timeseries = cutTimeSeriesIfNeeded(timeseries, self.BOLD_length,printInfo)[:self.N()]
        return {subjectID:
                    {'timeseries': timeseries,
                     'ABeta': abeta_burden,
                     'Tau': tau_burden,
                     'SC': SCnorm
                     }}

    def get_parcellation(self):
        return Glasser379.Glasser379()


# ================================================================================================================
#print('_Data_Raw loading done!')
# =========================  debug
if __name__ == '__main__':
    DL = ADNI_A()
    sujes = DL.get_classification()
    gCtrl = DL.get_groupSubjects('HC')
    s1 = DL.get_subjectData('002_S_0413')
    print('done! ;-)')
# ================================================================================================================
# ================================================================================================================
# ================================================================================================================EOF