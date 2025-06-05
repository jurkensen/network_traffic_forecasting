#!/bin/bash
#SBATCH --job-name=lstm_100_epochs      
#SBATCH --time=04:00:00                
#SBATCH --partition=gpu
#SBATCH --ntasks=1        
#SBATCH --gpus-per-task=1             
#SBATCH --cpus-per-task=8              
#SBATCH --mem-per-cpu=4G
#SBATCH --output=lstm_%j.out             
#SBATCH --error=lstm_%j.err              
#SBATCH --account=Education-EEMCS-Courses-CSE3000

# Display job info
echo "Job started at $(date)"
echo "Running on host: $(hostname)"

module load cuda/11.6

srun python -m models.LSTM

# Display job completion info
echo "Job completed at $(date)"