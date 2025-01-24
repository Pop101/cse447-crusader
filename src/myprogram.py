#!/usr/bin/env python
import os
import string
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from modules.simple_predictors import UniformRandomPredictor, WeightedRandomPredictor
from modules.dataloader import FixedLengthDataloader, NgramDataloader
from modules.normalizer import GutenbergNormalizer, StemmerNormalizer, TokenizerNormalizer
from modules.torchmodels import TransformerModel, CharTensorDataset, NgramCharTensorSet
from torch.utils.data import DataLoader
import torch

combined_normalizer = GutenbergNormalizer() + StemmerNormalizer() + TokenizerNormalizer()

TRAIN_DIR = '/job/data/data-train'
VAL_DIR   = '/job/data/data-val'

if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    random.seed(0)

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)
        
        print('Instatiating model')
        model = UniformRandomPredictor(string.ascii_letters)
        
        print('Loading training data')
        train_set = FixedLengthDataloader(TRAIN_DIR, fixed_length=100, overlap_size=10, skip_shorter_than=0, filters=[combined_normalizer])
        val_set   = FixedLengthDataloader(VAL_DIR,   fixed_length=100, overlap_size=10, skip_shorter_than=0, filters=[combined_normalizer])

        print('Training')
        model.run_train(train_set, args.work_dir)
        
        print('Saving model')
        model.save(args.work_dir)
        
    elif args.mode == 'test':
        print('Loading model')
        model = UniformRandomPredictor.load(args.work_dir)
        print('Loading test data from {}'.format(args.test_data))
        test_data = [] # Load data from here: args.test_data
        print('Making predictions')
        pred = model.run_pred(test_data)
        print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        model.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
