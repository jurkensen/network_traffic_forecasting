#!/bin/bash
#SBATCH --job-name=2min_cv_sarima      
#SBATCH --time=24:00:00    
#SBATCH --partition=compute            
#SBATCH --ntasks=1                     
#SBATCH --cpus-per-task=16             
#SBATCH --mem-per-cpu=3G
#SBATCH --output=cv_size_%j.out             
#SBATCH --error=cv_size_%j.err              
#SBATCH --account=Education-EEMCS-Courses-CSE3000

# Display job info
echo "Job started at $(date)"
echo "Running on host: $(hostname)"

srun python -m utils.cross

# Display job completion info
echo "Job completed at $(date)"