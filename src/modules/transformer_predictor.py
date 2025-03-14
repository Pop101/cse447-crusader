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
import math
from typing import Iterator, Tuple, List


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, max_seq_length, embed_size, num_heads, num_layers, dropout=0.1):
        super().__init__()
        self.max_seq_length = max_seq_length
        
        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_embedding = nn.Embedding(max_seq_length, embed_size)
        
        # Add layer normalization after embeddings
        self.layer_norm = nn.LayerNorm(embed_size)
        
        # Scale embeddings
        self.embed_scale = math.sqrt(embed_size)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dim_feedforward=embed_size * 4,  # Standard transformer uses 4x embed size
            dropout=dropout,
            batch_first=False  # Transformer expects [seq_len, batch, features]
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(embed_size)
        )
        
        self.fc = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def _generate_causal_mask(self, seq_len, device):
        """Generate a causal attention mask for the transformer"""
        # Create mask where each position can only attend to prior positions
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return mask
    
    def forward(self, x):
        # Add batch dimension if needed
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        batch_size, seq_len = x.shape
        
        # Find valid sequence lengths (non-padding tokens)
        # This helps us handle variable-length sequences properly
        seq_lengths = torch.sum(x > 0, dim=1)
        
        # Create padding mask (1 for padding, 0 for actual tokens)
        # For right-padded sequences, padding tokens are zeros at the end
        padding_mask = (x == 0)
        
        # Create position indices
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        # Apply token embeddings with scaling
        token_emb = self.token_embedding(x) * self.embed_scale
        
        # Apply positional embeddings
        pos_emb = self.pos_embedding(positions)
        
        # Combine embeddings and apply dropout
        x = self.dropout(token_emb + pos_emb)
        
        # Apply layer normalization
        x = self.layer_norm(x)
        
        # Transpose for transformer: [batch_size, seq_len, embed_size] -> [seq_len, batch_size, embed_size]
        x = x.transpose(0, 1)
        
        # Generate causal mask to prevent attending to future tokens
        # For right-padded sequences, this ensures tokens only attend to previous tokens
        causal_mask = self._generate_causal_mask(seq_len, x.device)
        
        # Apply transformer with both causal and padding masks
        # - causal_mask ensures each token only attends to previous tokens
        # - padding_mask ensures we ignore padding tokens completely
        x = self.transformer(
            x,
            mask=causal_mask,
            src_key_padding_mask=padding_mask
        )
        
        # Transpose back: [seq_len, batch_size, embed_size] -> [batch_size, seq_len, embed_size]
        x = x.transpose(0, 1)
        
        # Project to vocabulary size
        logits = self.fc(x)
        
        return logits
    
class TransformerPredictor(AbstractPredictor):
    def __init__(self, vocab_size, max_seq_length, embed_size, num_heads, num_layers, accumulation_steps=4) -> None:
        super().__init__()
        self.model = TransformerModel(vocab_size, max_seq_length, embed_size, num_heads, num_layers).to(device)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=0.001,  # Slightly lower learning rate for better stability
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
            ignore_index=0,  # Ignore padding token (0)
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
  
    def train_epoch(self, sequence_iterator:Iterator[torch.Tensor]) -> float:
        self.model.train()
        epoch_loss, epoch_batches = 0, 0
        
        self.optimizer.zero_grad()
        
        for i, sequences in enumerate(sequence_iterator):
            try:
                # sequences is a batch of sequences [batch_size, seq_len]
                batch_size, seq_len = sequences.size()
                
                # Create targets by shifting input one position left
                # For a sequence [a,b,c,d,0,0], the targets would be [b,c,d,0,0,0]
                targets = torch.zeros_like(sequences)
                if seq_len > 1:  # Only shift if sequence length > 1
                    targets[:, :-1] = sequences[:, 1:]
                
                # Forward pass - get predictions for each position
                output = self.model(sequences)  # [batch_size, seq_len, vocab_size]
                
                # Create a mask for positions where we have valid targets (not padding)
                # For causal LM, we only predict positions where the target is non-zero
                # and we don't predict for the last position (which has no next token)
                mask = (targets > 0)
                
                # We also don't want to include predictions for positions after valid sequence ends
                # First, get valid sequence lengths for each item in batch
                seq_lengths = torch.sum(sequences > 0, dim=1)
                
                # Adjust mask to only include positions within valid sequence length - 1
                # (since we're predicting the next token)
                valid_positions_mask = torch.zeros_like(mask, dtype=torch.bool)
                for b in range(batch_size):
                    # Only include positions up to valid sequence length - 1
                    # This ensures we're not trying to predict beyond the sequence
                    valid_length = max(0, min(seq_lengths[b].item() - 1, seq_len - 1))
                    valid_positions_mask[b, :valid_length] = True
                
                # Final mask is where we have both valid targets and valid positions
                final_mask = mask & valid_positions_mask
                
                # Reshape for loss calculation
                flat_output = output.reshape(-1, self.vocab_size)  # [batch_size*seq_len, vocab_size]
                flat_targets = targets.reshape(-1)                 # [batch_size*seq_len]
                flat_mask = final_mask.reshape(-1)                 # [batch_size*seq_len]
                
                # Skip batch if no valid predictions
                if not torch.any(flat_mask):
                    print("Warning: No valid prediction positions in batch, skipping")
                    continue
                
                # Select only the positions we want to predict
                masked_output = flat_output[flat_mask]
                masked_targets = flat_targets[flat_mask]
                
                # Safety check to ensure we're not going out of bounds
                if masked_targets.max() >= self.vocab_size:
                    print(f"Warning: Target indices exceed vocabulary size: {masked_targets.max()} >= {self.vocab_size}")
                    # Clamp targets to valid range
                    masked_targets = torch.clamp(masked_targets, 0, self.vocab_size - 1)
                
                # Calculate loss with gradient accumulation
                loss = self.criterion(masked_output, masked_targets) / self.accumulation_steps
                loss.backward()
                
                # Update weights after accumulation_steps
                if (i + 1) % self.accumulation_steps == 0:
                    # Clip gradients to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    torch.cuda.empty_cache()
                
                # Track statistics
                epoch_loss += loss.item() * self.accumulation_steps
                epoch_batches += 1
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "illegal memory" in str(e).lower():
                    print(f"WARNING: CUDA memory error, skipping batch {i}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    self.optimizer.zero_grad()  # Clear gradients to avoid accumulation
                    continue
                else:
                    print(f"Error in batch {i}: {str(e)}")
                    # Try to recover if possible
                    self.optimizer.zero_grad()
                    torch.cuda.empty_cache()
                    continue
                    
        # Handle any remaining gradients
        if epoch_batches % self.accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
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
        """
        Generate predictions for the next token after each sequence.
        
        Args:
            data: List of input tensors (sequences)
            temperature: Temperature for sampling (higher = more random)
            
        Returns:
            List of tensors containing top-k token predictions for each sequence
        """
        self.model.eval()
        preds = []
        
        with torch.no_grad():
            for item in data:
                try:
                    # Move to device
                    item = item.to(device)
                    
                    # Add batch dimension if needed
                    if item.dim() == 1:
                        item = item.unsqueeze(0)
                    
                    # Get model predictions
                    output = self.model(item)  # [batch_size, seq_len, vocab_size]
                    
                    # Find the last non-padding token position for each sequence
                    seq_lengths = torch.sum(item > 0, dim=1) - 1
                    
                    # Ensure we don't go out of bounds
                    seq_lengths = torch.clamp(seq_lengths, min=0, max=item.size(1) - 1)
                    
                    # Extract predictions for the last token in each sequence
                    batch_indices = torch.arange(output.size(0), device=output.device)
                    
                    # Safety check to avoid out-of-bounds errors
                    batch_indices = batch_indices[:output.size(0)]
                    seq_lengths = seq_lengths[:output.size(0)]
                    
                    # Get predictions for the last valid token in each sequence
                    last_token_output = output[batch_indices, seq_lengths]
                    
                    # Apply temperature scaling for sampling
                    scaled_logits = last_token_output / temperature
                    probs = torch.softmax(scaled_logits, dim=-1)
                    
                    # Get top-3 predictions
                    _, top_indices = torch.topk(probs, min(3, self.vocab_size), dim=-1)
                    
                    # Move to CPU for return
                    top_indices = top_indices.cpu()
                    
                    # Add to predictions list
                    preds.append(top_indices)
                    
                except Exception as e:
                    print(f"Warning: Error in prediction: {str(e)}")
                    # Add None as placeholder to maintain list length
                    preds.append(None)
                    continue
                    
        return preds

    def save(self, work_dir):
        # Create directory if it doesn't exist
        os.makedirs(work_dir, exist_ok=True)
        
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