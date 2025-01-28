import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm.auto import tqdm

class CharTensorDataset(Dataset):
    """
    A dataset that loads a list of strings,
    and returns two tensors where each element is a 0-X index of a character:
    - The first tensor is the string without the last character
    - The second tensor is the last character of the string
    """
    
    def __init__(self, strings):
        self.strings = list()
        self.seq_length = 0
        self.char_to_idx = {'<pad>': 0, '<break>': 1}
        self.idx_to_char = {0: '<pad>', 1: '<break>'}
        
        for string in tqdm(strings, desc="Building dataset..."):
            for char in string:
                if char not in self.char_to_idx:
                    idx = len(self.char_to_idx)
                    self.char_to_idx[char] = idx
                    self.idx_to_char[idx] = char
            self.seq_length = max(self.seq_length, len(string) - 1)
            self.strings.append(string)
    
    def __len__(self):
        return len(self.strings)
    
    def string_to_tensor(self, string):
        tensor = torch.ones(self.seq_length, dtype=torch.long) * self.char_to_idx['<pad>']
        for i, char in enumerate(string):
            tensor[i] = self.char_to_idx[char]
        return tensor
    
    def tensor_to_string(self, tensor):
        return "".join([self.idx_to_char[idx.item()] for idx in tensor])
    
    def __getitem__(self, idx):
        string = self.strings[idx]
        return self.string_to_tensor(string[:-1]), self.string_to_tensor(string[-1:])
    
class NgramCharTensorSet(Dataset):
    """
    A dataset that loads a list of ngrams (tuple of words),
    and returns two tensors where each element is a 0-X index of a character
    - The first tensor is the string without the last ngram
    - The second tensor is the last ngram
    """
    
    def __init__(self, ngrams):
        self.ngrams = list()
        self.seq_length = 0
        self.char_to_idx = {'<pad>': 0, '<break>': 1}
        self.idx_to_char = {0: '<pad>', 1: '<break>'}
        
        for ngram in tqdm(ngrams, desc="Building dataset..."):            
            # First, record all chars in ngram
            for onegram in ngram:
                for char in onegram:
                    if char not in self.char_to_idx:
                        idx = len(self.char_to_idx)
                        self.char_to_idx[char] = idx
                        self.idx_to_char[idx] = char
            
            # Now, record the length of the ngram
            x_gramlen = 0
            y_gramlen = 0
            for onegram in ngram[:-1]:
                if x_gramlen != 0:
                    x_gramlen += 1
                x_gramlen += len(onegram)
            for onegram in ngram[-1]:
                y_gramlen += len(onegram)                
            
            self.seq_length = max(self.seq_length, x_gramlen, y_gramlen)
            self.ngrams.append(ngram)
        
    def __len__(self):
        return len(self.ngrams)
    
    def ngram_to_tensor(self, ngram):
        tensor = torch.ones(self.seq_length, dtype=torch.long) * self.char_to_idx['<pad>']
        i = 0
        for onegram in ngram:
            if i > 0: # break only when needed
                tensor[i] = self.char_to_idx['<break>']
                i += 1
            for char in onegram:
                tensor[i] = self.char_to_idx[char]
                i += 1
        return tensor
    
    def tensor_to_ngram(self, tensor):
        ngram = []
        onegram = []
        for idx in tensor:
            char = self.idx_to_char[idx.item()]
            if char == '<break>':
                ngram.append("".join(onegram))
                onegram = []
            elif char == '<pad>':
                break # pad is always last char
            else:
                onegram.append(char)
        ngram.append("".join(onegram))
        return tuple(ngram)
    
    def __getitem__(self, idx):
        ngram = self.ngrams[idx]
        return self.ngram_to_tensor(ngram[:-1]), self.ngram_to_tensor(ngram[-1])
    

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_embedding = nn.Embedding(1000, embed_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_size, num_heads), num_layers
        )
        self.fc = nn.Linear(embed_size, vocab_size)
        
    def forward(self, x):
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        x = self.embedding(x) + self.pos_embedding(positions)
        mask = self._generate_padding_mask(x)
        x = self.transformer(x.permute(1, 0, 2), src_key_padding_mask=mask)
        return self.fc(x[-1])

    def _generate_padding_mask(self, x):
        # NOTE: this padding mask is dependent on the default <pad> token = 0
        # This is NO GOOD VERY BAD style
        return (x[:, :, 0] == 0).T

if __name__ == "__main__":
    ngrams = [
        ("hello", "world"),
        ("this", "is", "a", "test"),
        ("this", "is", "another", "test")
    ]
    dataset = NgramCharTensorSet(ngrams)
    print(dataset.char_to_idx)
    print(dataset.idx_to_char)
    print(dataset[0])
    print(dataset.tensor_to_ngram(dataset[0]))