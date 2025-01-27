from modules.abstract_predictor import AbstractPredictor
import random

import string
import random
import os

import json

class UniformRandomPredictor(AbstractPredictor):
    """
    This is a dummy model that predicts a random character each time.
    """

    def __init__(self, vocab:str = string.ascii_letters) -> None:
        super().__init__()
        self.vocab = set(vocab)
        
    def run_train(self, data, work_dir):
        self.vocab = set()
        for string in data:
            self.vocab.update(string)

    def run_pred(self, data):
        # your code here
        preds = []
        for input in data:
            # this model just predicts a random character each time
            top_guesses = [random.choice(self.vocab) for _ in range(3)]
            preds.append(''.join(top_guesses))
        return preds

    def save(self, work_dir):
        with open(os.path.join(work_dir, 'UniformRandomPredictor.checkpoint'), 'w') as f:
            f.write(''.join(self.vocab))

    @classmethod
    def load(cls, work_dir):
        with open(os.path.join(work_dir, 'UniformRandomPredictor.checkpoint')) as f:
            vocab = f.read()
        return UniformRandomPredictor(vocab=vocab)
    
class WeightedRandomPredictor(AbstractPredictor):
    """
    This is a dummy model that predicts a random character each time, weighted by the frequency of the character in the training data.
    """
    
    def __init__(self, vocab_weights:dict) -> None:
        super().__init__()
        self.vocab_weights = vocab_weights
        
    def run_train(self, data, work_dir):
        self.vocab_weights = {}
        for string in data:
            for char in string:
                if char not in self.vocab_weights:
                    self.vocab_weights[char] = 0
                self.vocab_weights[char] += 1
    
    def run_pred(self, data):
        preds = []
        for input in data:
            # this model just predicts a random character each time, weighted by occurance in source data
            top_guesses = random.choices(list(self.vocab_weights.keys()), weights=list(self.vocab_weights.values()), k=3)
            preds.append(''.join(top_guesses))
        return preds
    
    def save(self, work_dir):
        with open(os.path.join(work_dir, 'WeightedRandomPredictor.checkpoint'), 'wt') as f:
            json.dump(self.vocab_weights, f)
    
    @classmethod
    def load(cls, work_dir):
        with open(os.path.join(work_dir, 'WeightedRandomPredictor.checkpoint')) as f:
            vocab_weights = json.load(f)
        return WeightedRandomPredictor(vocab_weights=vocab_weights)