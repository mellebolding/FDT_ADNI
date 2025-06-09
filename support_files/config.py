import os

# Determine the root of the repo based on this file's location
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Main folders
scripts_dir = os.path.join(repo_root, 'python_scripts')
support_dir = os.path.join(repo_root, 'support_files')
dataloaders_dir = os.path.join(repo_root, 'DataLoaders')

# Data folders
adni_data_dir = os.path.join(repo_root, 'ADNI-A_DATA')
connectome_dir = os.path.join(adni_data_dir, 'connectomes')

# Optional additional folders
workbrain_dir = os.path.join(dataloaders_dir, 'WorkBrain')
raw_data_dir = os.path.join(workbrain_dir, '_Data_Raw')
produced_data_dir = os.path.join(workbrain_dir, '_Data_Produced')

# Print for debug (optional)
if __name__ == '__main__':
    print("Repo Root:", repo_root)
    print("Connectome Dir:", connectome_dir)
