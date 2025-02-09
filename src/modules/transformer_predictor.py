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
from typing import Iterator, Tuple

class TransformerPredictor(AbstractPredictor):
    def __init__(self, vocab_size, tensor_length, embed_size, num_heads, num_layers) -> None:
        super().__init__()
        self.model = TransformerModel(vocab_size, tensor_length, embed_size, num_heads, num_layers).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0.0001)
        self.criterion = nn.CrossEntropyLoss()
        
    def train_epoch(self, pair_iterator:Iterator[Tuple[torch.Tensor, torch.Tensor]]) -> float:
        self.model.train()
        total_loss = 0
        for X, y in pair_iterator:
            X = X.reshape(-1, X.size(-1)).to(device)  # Combine sequence and batch dimensions
            y = y.reshape(-1).to(device)              # Flatten targets
            
            self.optimizer.zero_grad()
            output = self.model(X)
            loss = self.criterion(output, y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss
    
    def run_pred(self, data):
        self.model.eval()
        with torch.no_grad():
            preds = []
            for item in data:
                try:
                    in_tensor = self.dataset.string_to_tensor(item)
                    in_tensor = in_tensor.unsqueeze(0).to(device)
                    output = self.model(in_tensor)
                    top_n_values, top_n_indices = torch.topk(output, 3, dim=-1)
                    top_n_chars = [self.dataset.idx_to_char[idx.item()] for idx in top_n_indices.squeeze()]
                    preds.append("".join(top_n_chars))
                except KeyError:
                    preds.append("".join(random.choices(string.ascii_letters, k=3)))
                    print(f"Warning: Unseen character in input \"{item}\". Replacing with random prediction.")
                    continue
        return preds

    def save(self, work_dir):
        with open(os.path.join(work_dir, 'model.pkl'), 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, work_dir):
        with open(os.path.join(work_dir, 'model.pkl'), 'rb') as f:
            return pickle.load(f)