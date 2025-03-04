#!/bin/bash

mkdir -p output
docker build -t cse447-proj/demo -f Dockerfile .

./container_run_testing.sh python src/myprogram.py test --work_dir work --test_data /job/data/example/input.txt --test_output ./output/pred.txt
python grader/grade.py pred.txt example/answer.txt --verbose