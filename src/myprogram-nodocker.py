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
from modules.datawriter import stream_to_single_parquet, stream_load_parquet
from modules.streamutil import chunker, sample_stream
from modules.pprint import TimerContext
from modules.torchgpu import device
import torch
import pandas as pd
from itertools import islice, chain
import pickle

combined_normalizer = GutenbergNormalizer() + StringNormalizer(remove_punct=False, lowercase=False)

# Data is located here:
# Leon's machine: '/mnt/e/data/gutenberg'
# Hyak: '/gscratch/gutenberg'

TRAIN_DIR = './data-train'
VAL_DIR   = './data-val'

limerator = lambda iter, max_n: map(lambda x: x[0], zip(iter, range(max_n)))

if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('prepare', 'process', 'train', 'test'), help='what to run')
    parser.add_argument('--data_dir', help='where to save', default='/gscratch/gutenberg')
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
        splitter  = SymlinkTestTrainSplit(args.data_dir, {
            TRAIN_DIR: 0.75,
            VAL_DIR: 0.25
        })
        splitter.split(random_state=42, verbose=True)
        
        
        print('Learning Vocab & Writing to .parquet')
        
        train_set = FixedLengthDataloader(TRAIN_DIR, fixed_length=100, overlap_size=10, skip_shorter_than=0, filters=[combined_normalizer])
        val_set   = FixedLengthDataloader(VAL_DIR,   fixed_length=100, overlap_size=10, skip_shorter_than=0, filters=[combined_normalizer])
        
        # Transform to chunks, then to pandas
        train_set = map(lambda x: pd.DataFrame(x), chunker(train_set, 1_000))
        val_set   = map(lambda x: pd.DataFrame(x), chunker(val_set, 1_000))
        
        # Learn vocab DURING iterator consume
        vocab = {'<PAD>': [0, 0], '<UNK>': [0, 0]}
        def learn_vocab(df):
            for char in chain.from_iterable(df['text'].values):
                if char not in vocab:
                    vocab[char] = [len(vocab), 1]
                else:
                    vocab[char][1] += 1
        
        train_set = map(lambda df: learn_vocab(df) or df, train_set)
        val_set   = map(lambda df: learn_vocab(df) or df, val_set)
        
        # Stream iterator to disk        
        train_file = os.path.join(args.work_dir, 'train.parquet')
        val_file   = os.path.join(args.work_dir, 'val.parquet')
        stream_to_single_parquet(train_set, train_file)
        stream_to_single_parquet(val_set, val_file)
        
        # Save vocab to disk
        with open(os.path.join(args.work_dir, 'vocab.pkl'), 'wb') as f:
            pickle.dump(vocab, f)
            
        print("Size of training set:\t{:.2f} MB".format(os.path.getsize(train_file) / 1e6))
        print("Size of validation set:\t{:.2f} MB".format(os.path.getsize(val_file) / 1e6))
    elif args.mode == 'process':
        # Process step converts training and validation data to tensors
        if not os.path.isdir(args.work_dir):
            print("Working directory {} does not exist".format(args.work_dir))
            exit(1)
        
        with TimerContext('Loading vocab'):
            with open(os.path.join(args.work_dir, 'vocab.pkl'), 'rb') as f:
                vocab = pickle.load(f)
            total_count = sum(freq for char, (idx, freq) in vocab.items())
            print(f"\tVocab contains {len(vocab)} unique characters ({total_count} total)")
            
        train_set         = stream_load_parquet(os.path.join(args.work_dir, 'train.parquet')) # Read from disk (too big for ram)
        train_set_texts   = chain.from_iterable(df['text'].values for df in train_set) # Select only text column, flatten
        train_set_tensors = stream_to_tensors(train_set_texts, 100, 128, lambda x: vocab.get(x, vocab['<UNK>'])[0]) # Convert to tensors w vocab
        train_set_tensors = map(lambda x: pd.DataFrame(x), chunker(train_set_tensors, 10_000))
        
        with TimerContext('Writing training tensors to disk'):
            stream_to_single_parquet(train_set_tensors, os.path.join(args.work_dir, 'train_tensors.parquet'))
        
        val_set         = stream_load_parquet(os.path.join(args.work_dir, 'val.parquet')) # Read from disk (too big for ram)
        val_set_texts   = chain.from_iterable(df['text'].values for df in val_set) # Select only text column, flatten
        val_set_tensors = stream_to_tensors(val_set_texts, 100, 128, lambda x: vocab.get(x, vocab['<UNK>'])[0]) # Convert to tensors w vocab
        val_set_tensors = map(lambda x: pd.DataFrame(x), chunker(val_set_tensors, 10_000))
        
        with TimerContext('Writing validation tensors to disk'):
            stream_to_single_parquet(val_set_tensors, os.path.join(args.work_dir, 'val_tensors.parquet'))
        
    elif args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print("Working directory {} does not exist".format(args.work_dir))
            exit(1)
        
        with TimerContext('Loading vocab'):
            with open(os.path.join(args.work_dir, 'vocab.pkl'), 'rb') as f:
                vocab = pickle.load(f)
            print(f"\tVocab contains {len(vocab)} characters")

        print('Instantiating model')
        model = TransformerPredictor(len(vocab), 100, 512, 8, 6)
        
        print('\nTraining model')
        for i_ in range(10):
            with TimerContext(f'Epoch {i_}'):
                # Build the iterator (pull-based streaming)
                train_set         = stream_load_parquet(os.path.join(args.work_dir, 'val.parquet')) # Read from disk (too big for ram)
                train_set_texts   = chain.from_iterable(df['text'].values for df in train_set) # Select only text column, flatten
                train_set_tensors = stream_to_tensors(train_set_texts, 100, 128, lambda x: vocab.get(x, vocab['<UNK>'])[0]) # Convert to tensors w vocab
                train_pairs       = create_random_length_sequence_pairs(train_set_tensors, 0, 100) # Create variable length sequences
                train_pairs       = sample_stream(train_pairs, 0.1) # Sample 10% of the data
                
                loss = model.train_epoch(train_pairs)
                print(f"Loss: {loss}")
                
                # Try to outrun oom (it wont work)
                del train_pairs, train_set_tensors, train_set_texts, train_set
        
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
