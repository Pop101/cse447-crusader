python src/myprogram-nodocker.py prepare --work_dir ./work --data_dir /mnt/e/data/multilingual && \
python src/myprogram-nodocker.py process --work_dir ./work 

sleep 1800 ; python src/myprogram-nodocker.py train --work_dir ./work --model transformer
sleep 1800 ; python src/myprogram-nodocker.py train --work_dir ./work --model rnn
