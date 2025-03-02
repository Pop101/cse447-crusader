python src/myprogram-nodocker.py prepare --work_dir ./work --data_dir /mnt/e/data/multilingual
python src/myprogram-nodocker.py process --work_dir ./work
python src/myprogram-nodocker.py train --work_dir ./work --model transformer
