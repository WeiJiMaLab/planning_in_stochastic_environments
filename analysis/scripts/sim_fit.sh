#!/bin/bash

# Example of running python script with a job array
#SBATCH -J model_fit
#SBATCH -c 1 # one CPU core per task
#SBATCH -t 3:00:00
#SBATCH --mem=4GB
#SBATCH --array=0-99
#SBATCH -o log/%j_%a.out
#SBATCH -e log/%j_%a.error 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hl3976@nyu.edu  
#SBATCH --job-name=TH-Final-Fit

# activate the venv
source /scratch/hl3976/ma_lab/.venv/bin/activate

# run the python script
python sim.py --type=$1