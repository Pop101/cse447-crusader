#!/usr/bin/env bash
set -x
set -e

rm -rf submit submit.zip
mkdir -p submit

# submit team.txt
printf "Leon Leibmann,lleibm\nNandini Talukdar,nandit\nKasten Welsh,kwelsh1" > submit/team.txt

# submit docker file
cp Dockerfile submit/Dockerfile

# submit source code
cp -r src submit/src

# Make sure we can run testing script
sudo ./container_run_testing.sh python src/myprogram.py test --work_dir work --test_data /job/data/example/input.txt --test_output submit/pred.txt

# submit checkpoints, ignoring .parquet & tensor.pt files (intermediate data)
mkdir -p submit/work
rsync -av --exclude='*.parquet' --exclude='*_tensors_*.pt' work/ submit/work/

# Submit files in root directory
find . -maxdepth 1 -type f ! -lname '*' ! -name '*.zip' -exec cp {} submit/ \;

# make zip file
zip -r Project447Group3.zip submit
