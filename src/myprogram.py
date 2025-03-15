#!/usr/bin/env python
import os
import string
import random
import json
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import torch
from train_transformer import CharDataset, CharTransformer, train_model, load_model, predict_next_char, top_3_predictions

class MyModel:
    """
    This is a starter model to get you started. Feel free to modify this file.
    """

    def __init__(self, seq_len=10):
        self.seq_len = seq_len
        self.model = None
        self.dataset = None
        self.char_to_idx = None
        self.idx_to_char = None

    @classmethod
    def load_training_data(cls, fname):
        with open(fname, 'r') as file:
            data = file.read()
        return data

    @classmethod
    def load_test_data(cls, fname):
        data = []
        with open(fname) as f:
            for line in f:
                inp = line.strip()  # Remove newline character
                data.append(inp)
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def run_train(self, data, work_dir):
        self.dataset = CharDataset(data, self.seq_len)
        self.model = CharTransformer(self.dataset.vocab_size)
        self.char_to_idx = self.dataset.char_to_idx
        self.idx_to_char = self.dataset.idx_to_char
        train_model(self.model, self.dataset, epochs=12, batch_size=16, lr=0.0005, save_path=os.path.join(work_dir, 'model.pth'))
        self.save_vocab(work_dir)

    def run_pred(self, data, work_dir):
        if self.model is None or self.dataset is None:
            self.load_vocab(work_dir)
            self.model = CharTransformer(len(self.char_to_idx))
            load_model(self.model, os.path.join(work_dir, 'model.pth'))
            self.dataset = CharDataset(''.join(data), self.seq_len)
            self.dataset.char_to_idx = self.char_to_idx
            self.dataset.idx_to_char = self.idx_to_char
        preds = []
        for inp in data:
            top3 = top_3_predictions(self.model, self.dataset, inp)
            preds.append(''.join(top3))  # Append the top 3 predicted characters
        return preds

    def save_vocab(self, work_dir):
        vocab = {
            'char_to_idx': self.char_to_idx,
            'idx_to_char': {int(k): v for k, v in self.idx_to_char.items()}  # Ensure keys are numbers
        }
        with open(os.path.join(work_dir, 'vocab.json'), 'w') as f:
            json.dump(vocab, f)

    def load_vocab(self, work_dir):
        with open(os.path.join(work_dir, 'vocab.json'), 'r') as f:
            vocab = json.load(f)
        self.char_to_idx = vocab['char_to_idx']
        self.idx_to_char = {int(k): v for k, v in vocab['idx_to_char'].items()}  # Ensure keys are numbers

    def save(self, work_dir):
        # Model saving is handled in run_train
        pass

    @classmethod
    def load(cls, work_dir):
        # Model loading is handled in run_pred
        return MyModel()


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--train_data', help='path to training data', default='example/train.txt')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    random.seed(0)

    model = MyModel()

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)
        print('Instantiating model')
        print('Loading training data from {}'.format(args.train_data))
        train_data = MyModel.load_training_data(args.train_data)
        print('Training')
        model.run_train(train_data, args.work_dir)
    elif args.mode == 'test':
        print('Loading model')
        model = MyModel.load(args.work_dir)
        print('Loading test data from {}'.format(args.test_data))
        test_data = MyModel.load_test_data(args.test_data)
        print('Making predictions')
        pred = model.run_pred(test_data, args.work_dir)
        print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        model.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
