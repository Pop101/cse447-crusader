from modules.abstract_predictor import AbstractPredictor
from modules.torchgpu import device
import torch
from torch import nn
import os
from typing import Iterator, Tuple, List
import math

class RNNWithAttentionModel(nn.Module):
    def __init__(self, vocab_size, max_seq_length, hidden_size, num_layers, num_heads=8, dropout=0.1):
        super().__init__()
        self.max_seq_length = max_seq_length
        
        # Ensure hidden_size is divisible by num_heads for multi-head attention
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_embedding = nn.Embedding(max_seq_length, hidden_size)
        
        # Normalization and scaling
        self.embed_norm = nn.LayerNorm(hidden_size)
        self.embed_scale = math.sqrt(hidden_size)
        
        # Bidirectional RNN
        self.rnn = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # Adjust for bidirectional output
        self.hidden_size_total = hidden_size * 2
        
        # Multi-head self-attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size_total,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.attn_norm = nn.LayerNorm(self.hidden_size_total)
        
        # Single unified projection network
        self.projection = nn.Sequential(
            nn.Linear(self.hidden_size_total, self.hidden_size_total * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size_total * 4, self.hidden_size_total * 2),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(self.hidden_size_total * 2, self.hidden_size_total),
            nn.LayerNorm(self.hidden_size_total)
        )
        
        # Output layer
        self.fc = nn.Linear(self.hidden_size_total, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        self.to(device)
    
    def forward(self, x):
        # Handle different input types (ensure proper tensor dimensions)
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension
            
        batch_size, seq_len = x.shape
        
        # Create padding mask (1 for padding positions, 0 for actual tokens)
        padding_mask = (x == 0)
        
        # Calculate valid sequence lengths for each item in batch
        seq_lengths = torch.sum(~padding_mask, dim=1).cpu()
        
        # Generate position indices
        positions = torch.arange(0, seq_len, device=x.device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)
        
        # Get embeddings with scaling
        word_embeddings = self.token_embedding(x) * self.embed_scale
        pos_embeddings = self.pos_embedding(positions)
        
        # Combine embeddings with dropout and normalization
        x = self.dropout(word_embeddings + pos_embeddings)
        x = self.embed_norm(x)
        
        # Pack padded sequence for more efficient RNN processing
        packed_x = nn.utils.rnn.pack_padded_sequence(
            x, seq_lengths, batch_first=True, enforce_sorted=False
        )
        
        # Process with RNN
        packed_output, (hidden, _) = self.rnn(packed_x)
        rnn_output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, total_length=seq_len
        )
        
        # Apply self-attention
        attn_output, _ = self.self_attention(
            query=rnn_output,
            key=rnn_output,
            value=rnn_output,
            key_padding_mask=padding_mask
        )
        
        attn_output = rnn_output + attn_output
        attn_output = self.attn_norm(attn_output)
        
        # Handle padding
        batch_indices = torch.arange(batch_size, device=x.device)
        last_indices = seq_lengths - 1
        last_indices = torch.clamp(last_indices, min=0)
        final_hidden = attn_output[batch_indices, last_indices]
        
        # Apply unified projection network
        projected = self.projection(final_hidden)
        logits = self.fc(projected)
        
        return logits


class RNNPredictor(AbstractPredictor):
    def __init__(self, vocab_size, max_seq_length, hidden_size, num_layers, num_heads=8) -> None:
        super().__init__()
        self.model = RNNWithAttentionModel(
            vocab_size=vocab_size, 
            max_seq_length=max_seq_length, 
            hidden_size=hidden_size, 
            num_layers=num_layers,
            num_heads=num_heads
        ).to(device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=0.001,
            betas=(0.9, 0.98),
            eps=1e-9,
            weight_decay=0.01
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )
        
        # Loss function with label smoothing
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=0,  # Ignore padding token
            label_smoothing=0.1  
        )
        
        # Store configuration
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        # Training metrics
        self.best_loss = float('inf')
        self.total_batches = 0
  
    def train_epoch(self, pair_iterator:Iterator[Tuple[torch.Tensor, torch.Tensor]]) -> float:
        self.model.train()
        epoch_loss, epoch_batches = 0, 0
        
        # Gradient accumulation steps
        accumulation_steps = 4
        
        for i, (X, y) in enumerate(pair_iterator):
            # Forward pass
            output = self.model(X)
            loss = self.criterion(output, y)
            
            # Scale loss for gradient accumulation
            scaled_loss = loss / accumulation_steps
            scaled_loss.backward()
            
            # Update weights every accumulation_steps
            if (i + 1) % accumulation_steps == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            # Track metrics
            epoch_loss += loss.item()
            epoch_batches += 1
        
        # Update scheduler once per epoch
        self.scheduler.step()
        
        # Update tracking metrics
        avg_loss = epoch_loss / epoch_batches if epoch_batches > 0 else float('inf')
        self.best_loss = min(self.best_loss, avg_loss)
        self.total_batches += epoch_batches
        
        return avg_loss
    
    def run_pred(self, data: List[torch.Tensor], temperature=1.0) -> List[torch.Tensor]:
        self.model.eval()
        preds = []
        
        with torch.no_grad():
            for item in data:
                try:
                    # Run model prediction
                    output = self.model(item)
                    
                    # Apply temperature scaling
                    scaled_logits = output / temperature
                    probs = torch.softmax(scaled_logits, dim=-1)
                    
                    # Get top predictions
                    top_n_values, top_n_indices = torch.topk(probs, 3, dim=-1)
                    
                    # Handle single item prediction
                    indices = top_n_indices.squeeze()
                    if indices.dim() == 0:
                        indices = indices.unsqueeze(0)
                    
                    # Add to predictions
                    preds.append(indices)
                except Exception as e:
                    preds.append(None)
                    print(f"Warning: Error in prediction: {str(e)}")
                    continue
                    
        return preds

    def save(self, work_dir):
        # Save model state and configuration
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'vocab_size': self.vocab_size,
            'max_seq_length': self.max_seq_length,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'best_loss': self.best_loss,
            'total_batches': self.total_batches
        }
        torch.save(state, os.path.join(work_dir, 'RNNPredictor.pt'))
            
    @classmethod
    def load(cls, work_dir):
        state = torch.load(os.path.join(work_dir, 'RNNPredictor.pt'), map_location=device)
        
        # Create predictor instance
        predictor = cls(
            state['vocab_size'],
            state['max_seq_length'], 
            state['hidden_size'],
            state['num_layers'],
            state.get('num_heads', 8)
        )
        
        # Load saved state
        predictor.model.load_state_dict(state['state_dict'])
        predictor.optimizer.load_state_dict(state['optimizer'])
        predictor.scheduler.load_state_dict(state['scheduler'])
        predictor.best_loss = state.get('best_loss', float('inf'))
        predictor.total_batches = state.get('total_batches', 0)
        
        return predictor