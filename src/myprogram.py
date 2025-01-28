#!/usr/bin/env python
import os
import string
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from tqdm.auto import tqdm

from modules.simple_predictors import UniformRandomPredictor, WeightedRandomPredictor
from modules.dataloader import FixedLengthDataloader, NgramDataloader, SymlinkTestTrainSplit
from modules.normalizer import GutenbergNormalizer, StemmerNormalizer, TokenizerNormalizer
from modules.torchmodels import TransformerModel, CharTensorDataset, NgramCharTensorSet
#from modules.torchgpu import device
import torch
import pandas as pd

combined_normalizer = GutenbergNormalizer() + StemmerNormalizer() + TokenizerNormalizer()

DATA_DIR  = '/job/data/data-all'
TRAIN_DIR = '/job/data/data-train'
VAL_DIR   = '/job/data/data-val'

limerator = lambda iter, max_n: map(lambda x: x[0], zip(iter, range(max_n)))

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

        train_set = limerator(train_set, 1000)
        val_set   = limerator(val_set, 1000)
        
        pd.DataFrame(train_set).to_pickle(os.path.join(args.work_dir, 'train.tar.gz'))
        pd.DataFrame(val_set).to_pickle(os.path.join(args.work_dir, 'val.tar.gz'))
        
        print("Data saved to {}".format(args.work_dir))
        print("Size of training set:\t{}mb".format(os.stat(os.path.join(args.work_dir, 'train.tar.gz')).st_size // 1024 // 1024))
        print("Size of validation set:\t{}mb".format(os.stat(os.path.join(args.work_dir, 'val.tar.gz')).st_size // 1024 // 1024))
        
    elif args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)
        
        print('Instatiating model')
        model = WeightedRandomPredictor(string.ascii_letters)
        
        print('Loading Data')
        train_set = pd.read_pickle(os.path.join(args.work_dir, 'train.tar.gz'))['text']
        val_set   = pd.read_pickle(os.path.join(args.work_dir, 'val.tar.gz'))['text']
        
        print("Learning Tensors")
        # train_set = CharTensorDataset(tqdm(train_set))
        # val_set   = CharTensorDataset(tqdm(val_set))

        print('Training')
        model.run_train(tqdm(train_set), args.work_dir)
        
        print('Saving model')
        model.save(args.work_dir)
        
    elif args.mode == 'test':
        print('Loading model')
        model = WeightedRandomPredictor.load(args.work_dir)
        
        print('Loading test data from {}'.format(args.test_data))
        test_data = []
        with open(args.test_data) as f:
            for line in f:
                test_data.append(line.strip())
        
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
