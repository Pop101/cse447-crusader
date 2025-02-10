# Model supplied by class

from typing import Iterator, Any, Tuple, List
import torch

class AbstractPredictor:
    """
    Abstract predictor interface we must extend for this assignment
    """

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def run_train(self, data:Any, work_dir:str):
        """Train on the given data, saving any necessary artifacts to work_dir"""
        raise NotImplementedError('This model cannot be trained')

    def train_epoch(self, pair_iterator:Iterator[Tuple[torch.Tensor, torch.Tensor]]) -> float:
        """Trains the model for one epoch, returning the loss"""
        raise NotImplementedError('This model cannot be trained per-epoch')
    
    def run_pred(self, data:List[str]) -> List[str]:
        """Predicts on the given data"""
        raise NotImplementedError('This model cannot be used for prediction')

    def save(self, work_dir:str):
        """Saves the model to work_dir"""
        raise NotImplementedError('This model cannot be saved')

    @classmethod
    def load(cls, work_dir:str):
        """Loads the model from work_dir"""
        raise NotImplementedError('This model cannot be loaded')