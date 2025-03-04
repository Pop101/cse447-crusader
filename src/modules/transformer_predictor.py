from modules.abstract_predictor import AbstractPredictor
from modules.torchmodels import CharTensorDataset, NgramCharTensorSet
import torch
from torch import nn
import os
import pickle
from tqdm.auto import tqdm
from modules.torchgpu import device
import random
import string
import math  # Added missing import
from typing import Iterator, Tuple, List


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, max_seq_length, embed_size, num_heads, num_layers, dropout=0.1):
        super().__init__()
        self.max_seq_length = max_seq_length
        
        # Increased embedding size for large vocabulary
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_embedding = nn.Embedding(max_seq_length, embed_size)
        
        # Add layer normalization after embeddings
        self.layer_norm = nn.LayerNorm(embed_size)
        
        # Scale embeddings
        self.embed_scale = math.sqrt(embed_size)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dim_feedforward=embed_size,
            dropout=dropout
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(embed_size)
        )
        
        self.fc = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def _generate_causal_mask(self, seq_len, device):
        """Generate a causal mask for the transformer model"""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return mask
    
    def forward(self, x):
        # Add batch dimension if needed
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        batch_size, seq_len = x.shape
        
        # Make padding mask for <PAD> tokens BEFORE reshape
        padding_mask = (x == 0).view(batch_size, -1)
        
        # Create position indices
        positions = torch.arange(0, seq_len, device=x.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)
        
        # Get scaled embeddings
        word_embeddings = self.token_embedding(x) * self.embed_scale
        pos_embeddings = self.pos_embedding(positions)
        
        # Combine and normalize embeddings
        x = self.dropout(word_embeddings + pos_embeddings)
        x = self.layer_norm(x)
        
        # Transformer expects shape: (seq_len, batch_size, embed_size)
        x = x.transpose(0, 1) 
        
        # Use both padding mask and causal mask
        x = self.transformer(
            x, 
            mask                 = self._generate_causal_mask(seq_len, x.device), # Causal mask
            src_key_padding_mask = padding_mask # Padding mask for <PAD> tokens
        )
        x = x.transpose(0, 1)
        # Back to (batch_size, seq_len, embed_size)
        
        # Project back to vocab size
        logits = self.fc(x)
        
        return logits
    
class TransformerPredictor(AbstractPredictor):
    def __init__(self, vocab_size, max_seq_length, embed_size, num_heads, num_layers, accumulation_steps=4) -> None:
        super().__init__()
        self.model = TransformerModel(vocab_size, max_seq_length, embed_size, num_heads, num_layers).to(device)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=0.01,
            betas=(0.9, 0.98),
            eps=1e-9,
            weight_decay=0.01
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=2,
            min_lr=1e-6
        )
        
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=0,  # Ignore padding token
            label_smoothing=0.1
        )
        
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        self.accumulation_steps = accumulation_steps
        
        self.best_loss = float('inf')
        self.total_batches = 0
  
    def train_epoch(self, pair_iterator:Iterator[Tuple[torch.Tensor, torch.Tensor]]) -> float:
        self.model.train()
        epoch_loss, epoch_batches = 0, 0
        
        accumulation_steps = 4  # Accumulate over 4 batches
        
        self.optimizer.zero_grad()
        
        for i, (X, y) in enumerate(pair_iterator):
            try:
                # Forward pass
                output = self.model(X)
                
                # Handle reshaping for loss calculation
                if output.dim() == 3:  # [batch_size, seq_len, vocab_size]
                    if y.dim() == 1:  # Single target per sequence
                        output = output[:, -1, :]
                    else:  # Target for each position
                        output = output.view(-1, self.vocab_size)
                        y = y.view(-1)
                
                loss = self.criterion(output, y) / accumulation_steps
                loss.backward()
                
                # Update weights every accumulation_steps
                if (i + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    torch.cuda.empty_cache()
                
                # Track statistics
                epoch_loss += loss.item() * accumulation_steps
                epoch_batches += 1
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "illegal memory" in str(e).lower():
                    print("WARNING: CUDA memory error, skipping batch")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    # Re-raise other errors
                    raise e
        
        # Handle any remaining gradients
        if epoch_batches % accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        # Calculate average loss
        avg_loss = epoch_loss / epoch_batches if epoch_batches > 0 else float('inf')
        
        # Update scheduler
        self.scheduler.step(avg_loss)
        self.best_loss = min(self.best_loss, avg_loss)
        self.total_batches += epoch_batches
        
        return avg_loss
    
    def run_pred(self, data: List[torch.Tensor], temperature=1.0) -> List[torch.Tensor]:
        self.model.eval()
        preds = []
        
        with torch.no_grad():
            for item in data:
                try:
                    item = item.to(device)
                    output = self.model(item)
                    
                    # Select the last token's output
                    if output.dim() == 3:  # [batch_size, seq_len, vocab_size]
                        # Get the last non-padding token
                        padding_mask = (item == 0)
                        last_non_pad = (~padding_mask).to(torch.long).sum(dim=1) - 1
                        last_non_pad = torch.clamp(last_non_pad, min=0)
                        
                        batch_indices = torch.arange(output.size(0), device=output.device)
                        output = output[batch_indices, last_non_pad]
                    
                    scaled_logits = output / temperature
                    probs = torch.softmax(scaled_logits, dim=-1)
                    
                    top_n_values, top_n_indices = torch.topk(probs, 3, dim=-1)
                    
                    # Handle single item prediction
                    indices = top_n_indices.squeeze()
                    if indices.dim() == 0:  # If only one prediction
                        indices = indices.unsqueeze(0)
                    
                    # Move back to CPU for further processing
                    indices = indices.cpu()
                    
                    # Dump indices into tensor
                    preds.append(indices)
                except Exception as e:
                    preds.append(None)
                    print(f"Warning: Error in prediction: {str(e)}")
                    continue
        return preds

    def save(self, work_dir):
        # Save the model state dict and other necessary components
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'vocab_size': self.vocab_size,
            'max_seq_length': self.max_seq_length,
            'embed_size': self.embed_size,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'best_loss': self.best_loss,
            'total_batches': self.total_batches
        }
        torch.save(state, os.path.join(work_dir, 'TransformerPredictor.pt'))
            

    @classmethod
    def load(cls, work_dir):
        state = torch.load(os.path.join(work_dir, 'TransformerPredictor.pt'), map_location=device)
        predictor = cls(
            state['vocab_size'],
            state['max_seq_length'], 
            state['embed_size'],
            state['num_heads'],
            state['num_layers']
        )
        
        predictor.model.load_state_dict(state['state_dict'])
        predictor.optimizer.load_state_dict(state['optimizer'])
        predictor.scheduler.load_state_dict(state['scheduler'])
        predictor.best_loss = state.get('best_loss', float('inf'))
        predictor.total_batches = state.get('total_batches', 0)
        return predictor