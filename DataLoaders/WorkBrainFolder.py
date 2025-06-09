import os
from pathlib import Path

# Get the directory of the script
base_folder = Path(__file__).resolve().parent  # Gets the current script's folder

# Define local subdirectories
WorkBrainFolder = base_folder / "WorkBrain"
WorkBrainDataFolder = WorkBrainFolder / "_Data_Raw"
WorkBrainProducedDataFolder = WorkBrainFolder / "_Data_Produced"

# Create directories if they don't exist
WorkBrainDataFolder.mkdir(parents=True, exist_ok=True)
WorkBrainProducedDataFolder.mkdir(parents=True, exist_ok=True)

#print("Base Folder:", WorkBrainFolder)
#print("Raw Data Folder:", WorkBrainDataFolder)
#print("Produced Data Folder:", WorkBrainProducedDataFolder)

parcellation_folder = WorkBrainDataFolder / '_Parcellations'