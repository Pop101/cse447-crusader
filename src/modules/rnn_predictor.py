from modules.abstract_predictor import AbstractPredictor
from modules.torchgpu import device
import torch
from torch import nn
import os
from typing import Iterator, List
import math


class RNNModel(nn.Module):
    def __init__(self, vocab_size, max_seq_length=100, hidden_size=256, num_layers=2, num_heads=4, dropout=0.1):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.hidden_size = hidden_size
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        
        # Positional encodings (fixed, not learned)
        position = torch.arange(0, max_seq_length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * -(math.log(10000.0) / hidden_size))
        pe = torch.zeros(max_seq_length, hidden_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
        # RNN layer
        self.rnn = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.attn_norm = nn.LayerNorm(hidden_size)
        
        # Projection network
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
        
        # Initialize parameters
        self._init_parameters()
        
        self.to(device)
    
    def _init_parameters(self):
        nn.init.normal_(self.token_embedding.weight, mean=0, std=0.02)
        for name, p in self.named_parameters():
            if "weight" in name and "norm" not in name and "embedding" not in name:
                if p.dim() >= 2:
                    nn.init.xavier_uniform_(p)
                else:
                    nn.init.normal_(p, mean=0, std=0.02)
    
    def forward(self, x):
        # Handle different input types
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension
            
        batch_size, seq_len = x.shape
        
        # Ensure sequence length doesn't exceed maximum allowed
        seq_len = min(seq_len, self.max_seq_length)
        if x.size(1) > self.max_seq_length:
            x = x[:, :self.max_seq_length]
        
        # Create padding mask (1 for padding positions, 0 for actual tokens)
        padding_mask = (x == 0)
        
        # Get embeddings 
        word_embeddings = self.token_embedding(x)
        
        # Add positional encodings
        positions = self.pe[:seq_len].unsqueeze(0)
        x = self.dropout(word_embeddings + positions)
        
        # Get sequence lengths for padding
        seq_lengths = torch.sum(~padding_mask, dim=1).cpu()
        seq_lengths = torch.clamp(seq_lengths, min=1)
        
        # Pack padded sequence for more efficient RNN processing
        packed_x = nn.utils.rnn.pack_padded_sequence(
            x, seq_lengths, batch_first=True, enforce_sorted=False
        )
        
        # RNN Pass
        packed_output, _ = self.rnn(packed_x)
        rnn_output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, total_length=seq_len
        )
        
        # Create causal attention mask (triangular mask)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        
        # Apply attention with causal mask
        attn_output, _ = self.attention(
            query=rnn_output,
            key=rnn_output,
            value=rnn_output,
            key_padding_mask=padding_mask,
            attn_mask=causal_mask
        )
        
        # Add residual connection and normalization
        attn_output = self.attn_norm(rnn_output + attn_output)
        
        # Apply projection to all sequence positions
        projected = self.projection(attn_output)
        
        # Get logits for all positions
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
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr           = 0.001,
            betas        = (0.9, 0.98),
            eps          = 1e-9,
            weight_decay = 0.01
        )
        
        self.scheduler = torch.optim.lr_scheduler.CyclicLR(
            self.optimizer,
            base_lr        = 1e-4,          # Min learning rate
            max_lr         = 0.01,          # Max learning rate
            step_size_up   = 2,             # Epochs to increase LR
            step_size_down = 5,             # Epochs to decrease LR
            mode           = 'triangular2', # Scaling policy (amplitude decreases each cycle)
            cycle_momentum = False          # Don't adjust momentum
        )
        
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
  
    def train_epoch(self, sequence_iterator: Iterator[torch.Tensor]) -> float:
        """
        Train the model for one epoch using causal language modeling.
        Each token tries to predict the next token in the sequence.
        """
        self.model.train()
        epoch_loss, epoch_batches = 0, 0
        
        # Zero gradients at the beginning
        self.optimizer.zero_grad(set_to_none=True)
        
        for i, sequences in enumerate(sequence_iterator):
            try:
                batch_size, seq_len = sequences.size()
                
                # Ensure sequence length doesn't exceed maximum allowed
                if seq_len > self.max_seq_length:
                    sequences = sequences[:, :self.max_seq_length]
                    seq_len = self.max_seq_length
                
                # Safety check - ensure no out-of-range indices
                if torch.max(sequences) >= self.vocab_size:
                    sequences = torch.clamp(sequences, 0, self.vocab_size - 1)
                
                # Move to device
                sequences = sequences.to(device)
                
                # Create target sequences (shifted input)
                input_seq = sequences
                target_seq = torch.zeros_like(sequences)
                
                if seq_len > 1:
                    # For causal LM: target is the next token for each position
                    target_seq[:, :-1] = sequences[:, 1:]
                
                # Forward pass
                logits = self.model(input_seq)
                
                # Simplified loss calculation:
                # 1. Reshape logits and targets for loss calculation
                flat_logits = logits.view(-1, self.vocab_size)
                flat_targets = target_seq.view(-1)
                
                # 2. Calculate the loss (CrossEntropyLoss will ignore padding tokens)
                loss = self.criterion(flat_logits, flat_targets)
                
                # Scale loss for gradient accumulation
                scaled_loss = loss / self.accumulation_steps
                scaled_loss.backward()
                
                # Update weights every accumulation_steps
                if (i + 1) % self.accumulation_steps == 0:
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                
                # Track metrics
                epoch_loss += loss.item()
                epoch_batches += 1
                
            except RuntimeError as e:
                print(f"Error in batch {i}: {str(e)}")
                torch.cuda.empty_cache()
                self.optimizer.zero_grad(set_to_none=True)
                continue
        
        # Handle any remaining gradients
        if epoch_batches % self.accumulation_steps != 0 and epoch_batches > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
        
        # Calculate average loss for scheduler
        avg_loss = epoch_loss / epoch_batches if epoch_batches > 0 else float('inf')
        self.scheduler.step(avg_loss)
        
        # Update tracking metrics
        self.best_loss = min(self.best_loss, avg_loss)
        self.total_batches += epoch_batches
        
        return avg_loss
    
    def run_pred(self, data: List[torch.Tensor], temperature=1.0) -> List[List[int]]:
        """
        Generate predictions for the next token after each sequence.
        """
        self.model.eval()
        preds = []
        
        with torch.no_grad():
            for item in data:
                try:
                    # Handle empty or None tensors
                    if item is None:
                        preds.append(None)
                        continue
                        
                    # Add batch dimension if needed
                    if item.dim() == 1:
                        item = item.unsqueeze(0)
                    
                    # Safety check for out-of-range indices
                    if torch.max(item) >= self.vocab_size:
                        item = torch.clamp(item, 0, self.vocab_size - 1)
                    
                    # Move to device
                    item = item.to(device)
                    
                    # Get model predictions
                    output = self.model(item)
                    
                    batch_size, seq_len, _ = output.size()
                    batch_predictions = []
                    
                    for b in range(batch_size):
                        # Find the last non-zero token position
                        non_zeros = (item[b] > 0).nonzero(as_tuple=True)[0]
                        
                        if len(non_zeros) > 0:
                            # Get the position of the last non-zero token
                            last_pos = min(non_zeros[-1].item(), seq_len - 1)
                            
                            # Get prediction for this position
                            logits = output[b, last_pos]
                            
                            # Apply temperature
                            scaled_logits = logits / temperature
                            probs = torch.softmax(scaled_logits, dim=-1)
                            
                            # Get top-k predictions
                            k = min(3, self.vocab_size)
                            _, top_indices = torch.topk(probs, k)
                            
                            # Convert to Python list
                            top_indices = top_indices.cpu().tolist()
                            batch_predictions.append(top_indices)
                        else:
                            # If no non-zero tokens, use first position
                            logits = output[b, 0]
                            scaled_logits = logits / temperature
                            probs = torch.softmax(scaled_logits, dim=-1)
                            _, top_indices = torch.topk(probs, min(3, self.vocab_size))
                            batch_predictions.append(top_indices.cpu().tolist())
                    
                    # Add batch predictions to overall predictions
                    if batch_predictions:
                        preds.append(batch_predictions[0] if len(batch_predictions) == 1 else batch_predictions)
                    else:
                        preds.append(None)
                    
                except Exception as e:
                    print(f"Warning: Error in prediction: {str(e)}")
                    preds.append(None)
                    continue
                    
        return preds

    def save(self, work_dir):
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