#!/bin/bash
#SBATCH --job-name=final_interval      
#SBATCH --time=24:00:00    
#SBATCH --partition=compute            
#SBATCH --ntasks=1                     
#SBATCH --cpus-per-task=32             
#SBATCH --mem-per-cpu=3G
#SBATCH --output=final_interval_%j.out             
#SBATCH --error=final_interval_%j.err              
#SBATCH --account=Education-EEMCS-Courses-CSE3000

# Display job info
echo "Job started at $(date)"
echo "Running on host: $(hostname)"

srun python -u main.py

# Display job completion info
echo "Job completed at $(date)"