from modules.abstract_predictor import AbstractPredictor
from modules.torchmodels import TransformerModel, CharTensorDataset, NgramCharTensorSet
import torch
from torch import nn
import os
import pickle
from tqdm.auto import tqdm
from modules.torchgpu import device
import random
import string
from typing import Iterator, Tuple, List

class TransformerPredictor(AbstractPredictor):
    def __init__(self, vocab_size, max_seq_length, embed_size, num_heads, num_layers) -> None:
        super().__init__()
        self.model = TransformerModel(vocab_size, max_seq_length, embed_size, num_heads, num_layers).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0.0001)
        self.criterion = nn.CrossEntropyLoss()
        
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        
    def train_epoch(self, pair_iterator:Iterator[Tuple[torch.Tensor, torch.Tensor]]) -> float:
        self.model.train()
        total_loss = 0
        for X, y in pair_iterator:
            self.optimizer.zero_grad()
            output = self.model(X)
            loss = self.criterion(output, y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss
    
    def run_pred(self, data: List[torch.Tensor], temperature=1.0) -> List[torch.Tensor]:
        self.model.eval()
        preds = []
        with torch.no_grad():
            for item in data:
                try:
                    # Model now handles 1D tensor input internally
                    output = self.model(item)
                    scaled_logits = output / temperature
                    probs = torch.softmax(scaled_logits, dim=-1)
                    
                    top_n_values, top_n_indices = torch.topk(output, 3, dim=-1)
                    
                    # Handle single item prediction
                    indices = top_n_indices.squeeze()
                    if indices.dim() == 0:  # If only one prediction
                        indices = indices.unsqueeze(0)
                    
                    # Dump indices into tensor
                    preds.append(indices)
                except KeyError as e:
                    preds.append(None)
                    print(f"Warning: Unseen character in input. Error: {str(e)}")
                    continue
        return preds

    def save(self, work_dir):
        # Save the model state dict and other necessary components
        state = {
            'state_dict': self.model.state_dict(),
            'vocab_size': self.vocab_size,
            'max_seq_length': self.max_seq_length,
            'embed_size': self.embed_size,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers
        }
        torch.save(state, os.path.join(work_dir, 'model.pt'))
            

    @classmethod
    def load(cls, work_dir):
        state = torch.load(os.path.join(work_dir, 'model.pt'), map_location=device)
        predictor = cls(
                state['vocab_size'],
                state['max_seq_length'], 
                state['embed_size'],
                state['num_heads'],
                state['num_layers']
        )
        
        predictor.model.load_state_dict(state['state_dict'])
        return predictor