#!/bin/bash -l
#SBATCH --job-name=cse447-model-train
#SBATCH --partition=ckpt-all
#SBATCH --account=bdata
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00
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
conda env create -f environment.yml -y
conda activate cse447

echo "
Env created!
--------------------
Checking for GPU...
"

python -c "import torch; print('CUDA is available.' if torch.cuda.is_available() else 'CUDA is not available.')"

echo "
--------------------
Preparing Data...
"

python src/myprogram-nodocker.py prepare --work_dir work --data_dir /gscratch/scrubbed/gutenberg

echo "
Done preparing!
--------------------
Training...
"

python src/myprogram-nodocker.py train --work_dir work --data_dir /gscratch/scrubbed/gutenberg
