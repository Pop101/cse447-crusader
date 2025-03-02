from modules.abstract_predictor import AbstractPredictor
from modules.torchgpu import device
import torch
from torch import nn
import os
from typing import Iterator, Tuple, List
import math
import warnings

class RNNModel(nn.Module):
    def __init__(self, vocab_size, max_seq_length=100, hidden_size=256, num_layers=2, num_heads=4, dropout=0.1):
        super().__init__()
        self.max_seq_length = max_seq_length
        
        # Reduced hidden size that scales better
        self.hidden_size = hidden_size
        
        # Embeddings with smaller dimensions
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        
        # Positional encodings (fixed, not learned)
        position = torch.arange(0, max_seq_length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * -(math.log(10000.0) / hidden_size))
        pe = torch.zeros(max_seq_length, hidden_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
        # Simplified RNN (use GRU instead of LSTM, unidirectional by default)
        self.rnn = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False  # Unidirectional for speed
        )
        
        # Attention mechanism (multi-head with dropout)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(hidden_size)
        
        # Simplified projection network (single layer with activation)
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size)
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize parameters with better defaults
        self._init_parameters()
        
        self.to(device)
    
    def _init_parameters(self):
        # Initialize embeddings and linear layers for faster convergence
        nn.init.normal_(self.token_embedding.weight, mean=0, std=0.02)
        for name, p in self.named_parameters():
            if "weight" in name and "norm" not in name and "embedding" not in name:
                if p.dim() >= 2:
                    nn.init.xavier_uniform_(p)
                else:
                    nn.init.normal_(p, mean=0, std=0.02)
    
    def forward(self, x):
        # Handle different input types (ensure proper tensor dimensions)
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension
            
        batch_size, seq_len = x.shape
        
        # Create padding mask (1 for padding positions, 0 for actual tokens)
        padding_mask = (x == 0)
        
        # Calculate valid sequence lengths for each item in batch
        seq_lengths = torch.sum(~padding_mask, dim=1).cpu()
        
        # Get embeddings 
        word_embeddings = self.token_embedding(x)
        
        # Add positional encodings, with dropout to not overfit
        positions = self.pe[:seq_len].unsqueeze(0)
        x = self.dropout(word_embeddings + positions)
        
        # Pack padded sequence for more efficient RNN processing
        packed_x = nn.utils.rnn.pack_padded_sequence(
            x, seq_lengths, batch_first=True, enforce_sorted=False
        )
        
        # RNN Pass
        packed_output, _ = self.rnn(packed_x)
        rnn_output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, total_length=seq_len
        )
        
        # Create causal attention mask (only looks back, not forward)
        seq_len = rnn_output.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        
        # Apply attention with causal mask
        attn_output, _ = self.attention(
            query=rnn_output,
            key=rnn_output,
            value=rnn_output,
            key_padding_mask=padding_mask,
            attn_mask=causal_mask  # Causal mask prevents attending to future tokens
        )
        
        # Add residual connection and normalization
        attn_output = rnn_output + attn_output
        
        # Get hidden state from last relevant position for each sequence
        batch_indices = torch.arange(batch_size, device=x.device)
        last_indices = seq_lengths - 1
        last_indices = torch.clamp(last_indices, min=0)
        final_hidden = attn_output[batch_indices, last_indices]
        final_hidden = self.attn_norm(final_hidden)
        
        # Apply simplified projection network
        projected = self.projection(final_hidden)
        logits = self.fc(projected)
        
        return logits


class RNNPredictor(AbstractPredictor):
    def __init__(self, vocab_size, max_seq_length, hidden_size, num_layers, num_heads=4, accumulation_steps=4) -> None:
        super().__init__()
        self.model = RNNModel(
            vocab_size=vocab_size, 
            max_seq_length=max_seq_length, 
            hidden_size=hidden_size, 
            num_layers=num_layers,
            num_heads=num_heads
        ).to(device)
        
        # Faster optimizer with reasonable defaults
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=0.001,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        )
        
        # Reduce learning rate on plateau
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=2,
            min_lr=1e-6
        )
        
        # Loss function (optional label smoothing)
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
        
        # Training parameters
        self.accumulation_steps = accumulation_steps
  
    def train_epoch(self, pair_iterator:Iterator[Tuple[torch.Tensor, torch.Tensor]]) -> float:
        self.model.train()
        epoch_loss, epoch_batches = 0, 0
        
        # Batched processing with mixed precision
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        
        for i, (X, y) in enumerate(pair_iterator):
            # Use automatic mixed precision for faster computation
            if scaler:
                with torch.cuda.amp.autocast():
                    # Forward pass
                    output = self.model(X)
                    loss = self.criterion(output, y)
                
                # Scale loss and gradients for mixed precision
                scaled_loss = loss / self.accumulation_steps
                scaler.scale(scaled_loss).backward()
                
                # Update weights every accumulation_steps
                if (i + 1) % self.accumulation_steps == 0:
                    # Clip gradients
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    # Optimizer step with scaling
                    scaler.step(self.optimizer)
                    scaler.update()
                    
                    self.optimizer.zero_grad(set_to_none=True)
            else:
                # Standard precision training (fallback)
                output = self.model(X)
                loss = self.criterion(output, y)
                
                scaled_loss = loss / self.accumulation_steps
                scaled_loss.backward()
                
                if (i + 1) % self.accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
            
            # Track metrics
            epoch_loss += loss.item()
            epoch_batches += 1
        
        # Calculate average loss for scheduler
        avg_loss = epoch_loss / epoch_batches if epoch_batches > 0 else float('inf')
        self.scheduler.step(avg_loss)  # Update scheduler based on validation loss
        
        # Update tracking metrics
        self.best_loss = min(self.best_loss, avg_loss)
        self.total_batches += epoch_batches
        
        return avg_loss
    
    def run_pred(self, data: List[torch.Tensor], temperature=1.0) -> List[torch.Tensor]:
        self.model.eval()
        preds = []
        
        with torch.no_grad():
            for item in data:
                try:
                    # Run model prediction with caching for efficiency
                    output = self.model(item)
                    
                    # Apply temperature scaling
                    scaled_logits = output / temperature
                    
                    # Get top predictions without computing all softmax values
                    top_n_values, top_n_indices = torch.topk(scaled_logits, 3, dim=-1)
                    
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
        # Save model state and configuration more efficiently
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
            state.get('num_heads', 4)
        )
        
        # Load saved state
        predictor.model.load_state_dict(state['state_dict'])
        predictor.optimizer.load_state_dict(state['optimizer'])
        predictor.scheduler.load_state_dict(state['scheduler'])
        predictor.best_loss = state.get('best_loss', float('inf'))
        predictor.total_batches = state.get('total_batches', 0)
        
        return predictor