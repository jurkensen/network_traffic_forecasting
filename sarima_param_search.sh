#!/bin/bash
#SBATCH --job-name=sarima_param_search      
#SBATCH --time=1:00:00                
#SBATCH --ntasks=1                     
#SBATCH --cpus-per-task=8              
#SBATCH --mem-per-cpu=1G
#SBATCH --partition=compute            
#SBATCH --output=sarima_param_search.out             
#SBATCH --error=sarima_param_search.err              
#SBATCH --account=Education-EEMCS-Courses-CSE3000

# Display job info
echo "Job started at $(date)"
echo "Running on host: $(hostname)"

module load py-pip
python -m pip install --user -r requirements.txt
srun python main.py

# Display job completion info
echo "Job completed at $(date)"