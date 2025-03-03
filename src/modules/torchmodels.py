from typing import Iterator, Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from modules.streamutil import chunker
from modules.torchgpu import device
import numpy as np
import math

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
        return self.string_to_tensor(string[:-1]), self.char_to_idx[string[-1]]

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
    

def stream_to_tensors(iterator, tensor_length:int, batch_size=1, vocab=ord, device=device) -> Iterator[torch.Tensor]:
    """
    Iterate over a stream of strings (or possibly ngrams), applying the "vocab" transformation
    to convert each string to a tensor.
    
    By default, the "vocab" is ord, the unicode value of the character.
    
    All tensors returned will be of tensor_length, padded with zeros.
    The iterator will be batched into batch_size.

    Args:
        iterator: iterator to be transformed
        tensor_length (int): length of the tensor. If space is not enough, the string will be truncated.
        batch_size (int, optional): Batch Size for the output. Defaults to 1.
        vocab (function->int, optional): Conversion function for the string. Defaults to ord.

    Returns:
        Iterator[torch.Tensor]: Stream of tensors
    """
    
    for chunk in chunker(iterator, batch_size):
        # Convert to tensors
        tensors = []
        for item in chunk:
            tensor = torch.zeros(tensor_length, dtype=torch.long)
            for i, char in enumerate(item):
                if i >= tensor_length:
                    break
                tensor[i] = vocab(char)
            tensors.append(tensor.to(device))
        
        # Stack tensors
        yield torch.stack(tensors)

def create_sequence_pairs(
    batched_tensors: Iterator[torch.Tensor],
    sequence_length: int,
):
    """
    Creates batched training pairs from pre-batched tensors for character-level

    Args:
        batched_tensors (Iterator[torch.Tensor]): Iterator of batched PyTorch tensors, each of shape (batch_size, sequence_length)
        sequence_length (int): Length of sequences to generate
    
    Yields:
        Tuple of (X, y) where:
            X: tensor of shape (sequence_length-1, batch_size) containing input sequences
            y: tensor of shape (1, batch_size) containing next character to predict
    """
    
    for batch in batched_tensors:
        X = batch[:, :-1]
        y = batch[:, -1:].squeeze(-1)
        yield (X, y)
        
def create_random_length_sequence_pairs(
    batched_tensors: Iterator[torch.Tensor],
    min_sequence_length: int,
    max_sequence_length: int,
):
    """
    Creates batched training pairs from pre-batched tensors for character-level.
    Sequences are of max_sequence_length, but masked to random lengths between min_sequence_length and max_sequence_length

    Args:
        batched_tensors (Iterator[torch.Tensor]): Iterator of batched PyTorch tensors, each of shape (batch_size, sequence_length)
        sequence_length (int): Length of sequences to generate
    
    Yields:
        Tuple of (X, y) where:
            X: tensor of shape (sequence_length-1, batch_size) containing input sequences
            y: tensor of shape (1, batch_size) containing next character to predict
    """
    for batch in batched_tensors:
        batch_size = batch.size(0)
        
        # Now transform to approximate normal distribution
        # Using Box-Muller transform on integers
        u1 = torch.rand(batch_size, device=batch.device)
        z = torch.sqrt(-2.0 * torch.log(u1)) * torch.cos(2.0 * torch.pi * torch.rand_like(u1))
        
        # Scale and shift to desired range
        mean = (min_sequence_length + max_sequence_length) / 2
        std = (max_sequence_length - min_sequence_length) / 6
        lengths = ((z * std) + mean).long()
        lengths = torch.clamp(lengths, min=min_sequence_length, max=max_sequence_length)
        
        # Create position indices tensor
        positions = torch.arange(max_sequence_length-1, device=batch.device).expand(batch_size, -1)
        
        # Create mask where positions are less than effective_lengths
        mask = positions < lengths.unsqueeze(1)
        
        # Get X and apply mask
        X = batch[:, :-1] * mask
        
        # Get y values at effective_lengths positions
        y = torch.gather(batch, 1, lengths.unsqueeze(1)).squeeze(1)
        
        yield (X, y)

def create_variable_length_sequence_pairs(
    batched_tensors: Iterator[torch.Tensor],
    max_sequence_length: int,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Creates batched training pairs from pre-batched tensors for character-level language modeling.
    Sequences are variable length, with padding added for all variations from 0..max_sequence_length
    
    Args:
        batched_tensors: Iterator of batched PyTorch tensors, each of shape (batch_size, sequence_length)
        max_sequence_length: Maximum length of output sequences
        device: Target device for tensors
        
    Yields:
        Tuple of (X, y) where:
            X: tensor of shape (sequence_length-1, batch_size) containing input sequences
            y: tensor of shape (1, batch_size) containing next character to predict
    """
    for batch in batched_tensors:
        sequence_length = batch.size(1)
        batch_size = batch.size(0)
        
        # Preallocate tensors for all sequences in this batch
        all_X = []
        all_y = []
        
        # Generate all sequences for current batch
        for end_idx in range(1, sequence_length):
            # Input sequences up to current position
            X = batch[:, :end_idx].clone()
            
            # Target is next character for each sequence in batch
            y = batch[:, end_idx].clone()
            
            # Pad if needed
            if X.size(1) < max_sequence_length:
                pad_size = max_sequence_length - X.size(1)
                X = torch.nn.functional.pad(X, (0, pad_size), value=0)
            elif X.size(1) > max_sequence_length:
                X = X[:, :max_sequence_length]
            
            all_X.append(X)
            all_y.append(y)
        
        # Stack all sequences for this batch
        X_batch = torch.stack(all_X)  # shape: (sequence_length-1, batch_size, max_sequence_length)
        y_batch = torch.stack(all_y)  # shape: (sequence_length-1, batch_size)
        
        yield X_batch, y_batch

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