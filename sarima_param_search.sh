#!/bin/bash
#SBATCH --job-name=sarima_param_search      
#SBATCH --time=36:00:00    
#SBATCH --partition=compute            
#SBATCH --ntasks=1                     
#SBATCH --cpus-per-task=16              
#SBATCH --mem-per-cpu=4G
#SBATCH --output=sarima_param_search_%j.out             
#SBATCH --error=sarima_param_search_%j.err              
#SBATCH --account=Education-EEMCS-Courses-CSE3000

# Display job info
echo "Job started at $(date)"
echo "Running on host: $(hostname)"

srun python -u main.py

# Display job completion info
echo "Job completed at $(date)"