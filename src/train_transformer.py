import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Check for MPS (Mac GPU) or CUDA
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Character-level dataset
class CharDataset(Dataset):
    def __init__(self, text, seq_len):
        chars = sorted(list(set(text)))
        self.unk_token = "<UNK>"
        chars = [self.unk_token] + chars
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)
        self.seq_len = seq_len
        self.data = [self.char_to_idx[ch] for ch in text]

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        input_seq = self.data[idx:idx + self.seq_len]
        target = self.data[idx + self.seq_len]
        return torch.tensor(input_seq), torch.tensor(target)

# Transformer Model
class CharTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=8, num_layers=3, max_seq_len=100):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional Encoding
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(d_model, vocab_size)

    def generate_causal_mask(self, size):
        """Generate a causal mask for self-attention"""
        mask = torch.triu(torch.ones(size, size), diagonal=1)  # Upper triangular mask
        mask = mask.masked_fill(mask == 1, float('-inf'))  # Convert to -inf for masking
        return mask.to(device)

    def forward(self, x):
        x = self.embedding(x)  # (batch, seq_len, d_model)
        seq_len = x.shape[1]
        x = x + self.positional_encoding[:, :seq_len, :]  # Add positional encoding
        x = x.permute(1, 0, 2)  # (seq_len, batch, d_model) for Transformer

        # Apply causal mask
        causal_mask = self.generate_causal_mask(seq_len)
        x = self.transformer(x, mask=causal_mask)

        x = x[-1]  # Take last output for prediction
        return self.fc(x)


# Training and Testing Functions
def train_model(model, dataset, epochs=10, batch_size=32, lr=0.001, save_path="model.pth"):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    for epoch in range(epochs):
        total_loss = 0
        for inputs, targets in tqdm(dataloader, desc=f"Epoch {epoch + 1}"):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")
    
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

def load_model(model, load_path="model.pth"):
    model.load_state_dict(torch.load(load_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded from {load_path}")

def predict_next_char(model, dataset, start_text, predict_len=1):
    model.eval()
    input_seq = [dataset.char_to_idx[ch] if ch in dataset.char_to_idx else dataset.char_to_idx[dataset.unk_token] for ch in start_text[-dataset.seq_len:]]
    input_seq = torch.tensor(input_seq).unsqueeze(0).to(device)  # Add batch dimension
    with torch.no_grad():
        for _ in range(predict_len):
            output = model(input_seq)
            predicted_idx = torch.argmax(output, dim=-1).item()
            start_text += dataset.idx_to_char[predicted_idx]
            input_seq = torch.cat([input_seq[:, 1:], torch.tensor([[predicted_idx]]).to(device)], dim=1)
    return start_text

def top_3_predictions(model, dataset, start_text):
    model.eval()
    input_seq = [dataset.char_to_idx[ch] if ch in dataset.char_to_idx else dataset.char_to_idx[dataset.unk_token] for ch in start_text[-dataset.seq_len:]]
    input_seq = torch.tensor(input_seq).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_seq)
        probabilities = torch.softmax(output, dim=-1)
        top5_indices = torch.topk(probabilities, 5, dim=-1).indices.squeeze().tolist()
        top3_indices = torch.topk(probabilities, 3, dim=-1).indices.squeeze().tolist()
        top5_chars = [dataset.idx_to_char[idx] for idx in top5_indices]
        top3_chars = [dataset.idx_to_char[idx] for idx in top3_indices]
        print(f'Start Text: {start_text}')
        print("Predicted Text: ", predict_next_char(model, dataset, start_text, predict_len=10))
        print(top3_chars)
        #  if one of the top 3 is a new line or <unk>, replace it with the next highest probability
        j = 4
        for i in top3_chars:
            if i == '\n' or i == '<UNK>':
                top3_chars.remove(i)
                top3_chars.append(dataset.idx_to_char[top5_indices[j]])
                j += 1
    return top3_chars

# # Example Usage
# with open("/Users/kastenwelsh/Documents/cse447-project/src/minispeare.txt", "r") as file:
#     text = file.read()

# dataset = CharDataset(text, seq_len=10)
# model = CharTransformer(dataset.vocab_size)
# train_model(model, dataset, epochs=20, batch_size=16, lr=0.0005)

# # Save and Load Model
# torch.save(model.state_dict(), "char_transformer.pth")
# model.load_state_dict(torch.load("char_transformer.pth", map_location=device))
# model.to(device)
# model.eval()

# # Testing
# start_seq = "AUFIDIUS. "
# print(predict_next_char(model, dataset, start_seq, predict_len=100))

# print("Top 3 predictions:", top_3_predictions(model, dataset, start_seq))