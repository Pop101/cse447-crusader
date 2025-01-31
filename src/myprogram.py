#!/usr/bin/env python
import os
import string
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from tqdm.auto import tqdm

from modules.simple_predictors import UniformRandomPredictor, WeightedRandomPredictor
from modules.dataloader import FixedLengthDataloader, NgramDataloader, SymlinkTestTrainSplit
from modules.normalizer import GutenbergNormalizer, StemmerNormalizer, TokenizerNormalizer
from modules.torchmodels import CharTensorDataset, NgramCharTensorSet
from modules.transformer_predictor import TransformerPredictor

from modules.torchgpu import device
import torch
import pandas as pd
from itertools import islice

combined_normalizer = GutenbergNormalizer() + StemmerNormalizer() + TokenizerNormalizer()

DATA_DIR  = '/job/data/data-all'
TRAIN_DIR = '/job/data/data-train'
VAL_DIR   = '/job/data/data-val'

limerator = lambda iter, max_n: map(lambda x: x[0], zip(iter, range(max_n)))
chunker = lambda it, chunk_size: iter(lambda: list(islice(it, chunk_size)), [])

if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('prepare', 'train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    random.seed(0)
    
    #print("Running in {}".format(device))

    if args.mode == 'prepare':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)
            
        print('Performing Test/Train Split')
        splitter  = SymlinkTestTrainSplit(DATA_DIR, {
            TRAIN_DIR: 0.75,
            VAL_DIR: 0.25
        })
        splitter.split(random_state=42)
        
        train_set = FixedLengthDataloader(TRAIN_DIR, fixed_length=100, overlap_size=10, skip_shorter_than=0, filters=[combined_normalizer])
        val_set   = FixedLengthDataloader(VAL_DIR,   fixed_length=100, overlap_size=10, skip_shorter_than=0, filters=[combined_normalizer])

        train_set = limerator(train_set, 1000_000)
        val_set   = limerator(val_set, 1000_000)
        
        # TODO: stream iterator to disk
        output_path = os.path.join(args.work_dir, 'train.parquet')
        first_chunk = True
        for chunk in tqdm(chunker(train_set, 10_000)):
            df = pd.DataFrame({'text': chunk})
            if first_chunk:
                df.to_parquet(output_path, compression='snappy')
                first_chunk = False
            else:
                df.to_parquet(output_path, compression='snappy', append=True)
        
        print(pd.read_pickle(os.path.join(args.work_dir, output_path)))
        
        print("Data saved to {}".format(args.work_dir))
        print("Size of training set:\t{}mb".format(os.stat(os.path.join(args.work_dir, output_path)).st_size // 1024 // 1024))
        
    elif args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)
        
        print('Instatiating model')
        model = TransformerPredictor()
        
        print('Loading Data')
        train_set = pd.read_pickle(os.path.join(args.work_dir, 'train.tar.gz'))['text']
        val_set   = pd.read_pickle(os.path.join(args.work_dir, 'val.tar.gz'))['text']

        print('Training')
        model.run_train(train_set, args.work_dir)
        
        print('Saving model')
        model.save(args.work_dir)
        
    elif args.mode == 'test':
        print('Loading model')
        model = TransformerPredictor.load(args.work_dir)
        
        print('Loading test data from {}'.format(args.test_data))
        test_data = []
        with open(args.test_data) as f:
            for line in f:
                test_data.append(combined_normalizer(line))

        
        print('Making predictions')
        pred = model.run_pred(test_data)
        
        print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        model.write_pred(pred, args.test_output)
    
    elif args.mode == 'tui':
        print('Loading model')
        model = WeightedRandomPredictor.load(args.work_dir)
        
        while True:
            input = input('Input: ')
            pred = model.run_pred([input])
            print(pred)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
