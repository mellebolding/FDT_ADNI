#!/bin/bash
#SBATCH --job-name=EC_sigma_fit_v2         # Job name
#SBATCH --output=EC_sigma_fit_v2.out       # Standard output log
#SBATCH --error=EC_sigma_fit_v2.err        # Standard error log
#SBATCH --time=300:00:00                 # Time limit (hh:mm:ss)
#SBATCH --ntasks=1                      # Number of tasks (processes)
#SBATCH --cpus-per-task=1               # Number of CPU cores per task
#SBATCH --mem=2G                        # Memory per node (adjust as needed)
#SBATCH --mail-type=END,FAIL            # Notifications for job done or fail (optional)

# Activate your virtual environment
source /home/cluster/melle/FDT_ADNI/venv/bin/activate

export MPLCONFIGDIR=/tmp/mplconfig_$SLURM_JOB_ID
mkdir -p $MPLCONFIGDIR

# Go to your script directory (optional but clean)
cd /home/cluster/melle/FDT_ADNI/python_scripts

# Run your Python script
python EC_sigma_fit_v2.py
