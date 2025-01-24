from modules.abstract_predictor import AbstractPredictor
from modules.torchmodels import TransformerModel, CharTensorDataset, NgramCharTensorSet
import torch
import os
import pickle

class CharTensorDataset(AbstractPredictor):
    def __init__(self) -> None:
        super().__init__()
        self.model = None
        
    def run_train(self, data, work_dir):
        self.dataset = CharTensorDataset(data)
        self.model = TransformerModel(
            vocab_size=len(self.dataset.char_to_idx),
            embed_size=512,
            num_heads=8,
            num_layers=6
        )
        
        train_set_loader = torch.utils.data.DataLoader(self.dataset, batch_size=32, shuffle=True)

        # Train the model
        num_epochs = 10
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        criterion = torch.nn.CrossEntropyLoss()
        for epoch in range(num_epochs):
            self.model.train()
            for batch in train_set_loader:
                input, target = batch
                optimizer.zero_grad()
                
                # Forward
                output = self.model(input)
                loss = criterion(output.view(-1, output.size(-1)), target.view(-1))
                
                # Backward
                loss.backward()
                optimizer.step()

    def run_pred(self, data):
        self.model.eval()
        with torch.no_grad():
            preds = []
            for item in data:
                in_tensor = self.dataset.ngram_to_tensor(item)
                output = self.model(in_tensor)
                print(output)
                top_n_values, top_n_indices = torch.topk(output, 3, dim=-1)
        return preds

    def save(self, work_dir):
        with open(os.path.join(work_dir, 'model.pkl'), 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, work_dir):
        with open(os.path.join(work_dir, 'model.pkl'), 'rb') as f:
            return pickle.load(f)