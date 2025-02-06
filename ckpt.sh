#!/bin/bash
#SBATCH --job-name=data-proc
#SBATCH --partition=ckpt
#SBATCH --account=stf
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kwelsh1@uw.edu
# I use source to initialize conda into the right environment.
source ~/.bashrc
cat $0
echo "
--------------------
"
bash script.sh arg1 arg2