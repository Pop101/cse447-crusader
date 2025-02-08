#!/bin/bash -l
#SBATCH --job-name=cse447-model-train
#SBATCH --partition=ckpt-all
#SBATCH --account=bdata
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --gres=gpu:1
#SBATCH --time=999:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=lleibm@uw.edu

# I use source to initialize conda into the right environment.
shopt -s expand_aliases
source ~/.bashrc
eval "$(micromamba shell hook --shell=bash)"

echo "
--------------------
Setting Conda Env:
"

# Check if the conda environment exists
if ! conda info --envs | awk '{print $1}' | grep -qx "cse447-v2"; then
    echo "Conda environment 'cse447' not found. Creating it..."
    conda env create -f environment.yml -y
else
    echo "Conda environment 'cse447' already exists. Skipping creation."
fi

# Activate environment
conda activate cse447-v2

echo "
Env setup complete!
--------------------
Checking for GPU...
"

python -c "import torch; print('CUDA is available.' if torch.cuda.is_available() else 'CUDA is not available.')"

echo "
--------------------
Preparing Data...
"

if [[ ! -f ./work/train.parquet ]]; then
    echo "train.parquet not found. Running data preparation..."
    python src/myprogram-nodocker.py prepare --work_dir work --data_dir /gscratch/scrubbed/gutenberg
    echo "Done preparing!"
else
    echo "train.parquet already exists. Skipping data preparation."
fi

echo "
--------------------
Training...
"

python src/myprogram-nodocker.py train --work_dir work --data_dir /gscratch/scrubbed/gutenberg
