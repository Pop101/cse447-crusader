#!/usr/bin/env bash
set -x
set -e

rm -rf submit submit.zip
mkdir -p submit

# submit team.txt
printf "Leon Leibmann,lleibm\nNandini Talukdar,nandit\nKasten Welsh,kwelsh1" > submit/team.txt

# train model (takes FOREVER)
python src/myprogram.py train --work_dir work

# make predictions on example data submit it in pred.txt
python src/myprogram.py test --work_dir work --test_data example/input.txt --test_output submit/pred.txt

# submit docker file
cp Dockerfile submit/Dockerfile

# submit source code
cp -r src submit/src

# submit checkpoints
cp -r work submit/work

# make zip file
zip -r submit.zip submit
