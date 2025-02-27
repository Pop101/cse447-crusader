#!/usr/bin/env python
import os
import string
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from tqdm.auto import tqdm

from modules.simple_predictors import UniformRandomPredictor, WeightedRandomPredictor
from modules.dataloader import FixedLengthDataloader, NgramDataloader, SymlinkTestTrainSplit
from modules.normalizer import GutenbergNormalizer, StemmerNormalizer, TokenizerNormalizer, StringNormalizer
from modules.torchmodels import CharTensorDataset, NgramCharTensorSet, stream_to_tensors, create_sequence_pairs, create_random_length_sequence_pairs
from modules.transformer_predictor import TransformerPredictor
from modules.rnn_predictor import RNNPredictor
from modules.datawriter import stream_to_single_parquet, stream_load_parquet, stream_load_pt_glob
from modules.streamutil import chunker, sample_stream
from modules.pprint import TimerContext
from modules.torchgpu import device
import torch
import pandas as pd
from itertools import islice, chain
import pickle

combined_normalizer = GutenbergNormalizer() + StringNormalizer(remove_punct=False, lowercase=False)

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
        raise NotImplementedError('Prepare mode not supported with docker')
        
    elif args.mode == 'train':
        raise NotImplementedError('Train mode not supported with docker')

    elif args.mode == 'test':
        with TimerContext('Loading vocab'):
            with open(os.path.join(args.work_dir, 'vocab.pkl'), 'rb') as f:
                vocab = pickle.load(f)
            vocab_list = list(vocab.keys())
            print(f"\tVocab contains {len(vocab)} characters")
            
        with TimerContext('Loading model'):
            model = RNNPredictor.load(args.work_dir)
            print(f"\tModel loaded, total batches: {model.total_batches}")
        
        print('Loading test data from {}'.format(args.test_data))
        test_data = []
        with open(args.test_data) as f:
            for line in f:
                norm_line = combined_normalizer(line)
                norm_line = norm_line[-99:] if len(norm_line) > 99 else norm_line
                test_data.append(combined_normalizer(line))
                
        test_data_stream = stream_to_tensors(test_data, 99, 1, lambda x: vocab.get(x, vocab['<UNK>'])[0])
        test_data_stream = map(lambda x: x.to(device).squeeze(0), test_data_stream)
        
        print('Making predictions')
        pred_tensors = model.run_pred(test_data_stream)
        pred = ["".join([vocab_list[i] for i in p]) if p != None else 'xxx' for p in pred_tensors]
        
        print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        model.write_pred(pred, args.test_output)
    
    elif args.mode == 'tui':
        with TimerContext('Loading vocab'):
            with open(os.path.join(args.work_dir, 'vocab.pkl'), 'rb') as f:
                vocab = pickle.load(f)
            vocab_list = list(vocab.keys())
            print(f"\tVocab contains {len(vocab)} characters")
            
        with TimerContext('Loading model'):
            model = TransformerPredictor.load(args.work_dir)
        
        while True:
            line = input('Input: ')
            
            norm_line = combined_normalizer(line)
            norm_line = norm_line[-99:] if len(norm_line) > 99 else norm_line
            test_data = [combined_normalizer(norm_line)]
            
            test_data_stream = stream_to_tensors(test_data, 99, 1, lambda x: vocab.get(x, vocab['<UNK>'])[0])
            test_data_stream = map(lambda x: x.to(device).squeeze(0), test_data_stream)
            
            pred_tensors = model.run_pred(test_data_stream)
            pred = ["".join([vocab_list[i] for i in p]) if p != None else 'xxx' for p in pred_tensors]
            
            print('Prediction: {}'.format(pred[0]))
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
