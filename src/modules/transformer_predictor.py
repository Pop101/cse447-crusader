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
        
        # IMPORTANT FIX 1: Ensure sequence length doesn't exceed maximum allowed
        seq_len = min(seq_len, self.max_seq_length)
        
        # Safety check - ensure no out-of-range indices
        if torch.max(x) >= self.token_embedding.num_embeddings:
            print(f"Warning: Input contains indices >= vocab_size ({torch.max(x).item()} >= {self.token_embedding.num_embeddings})")
            # Clamp values to valid range
            x = torch.clamp(x, 0, self.token_embedding.num_embeddings - 1)
        
        # Find valid sequence lengths (non-padding tokens)
        seq_lengths = torch.sum(x > 0, dim=1)
        seq_lengths = torch.clamp(seq_lengths, min=0, max=seq_len)
        
        # Create padding mask (1 for padding, 0 for actual tokens)
        padding_mask = (x == 0)
        
        # Create position indices
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        # IMPORTANT FIX 2: Ensure position indices are within bounds
        positions = torch.clamp(positions, 0, self.pos_embedding.num_embeddings - 1)
        
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
        causal_mask = self._generate_causal_mask(seq_len, x.device)
        
        # Apply transformer with both causal and padding masks
        try:
            x = self.transformer(
                x,
                mask=causal_mask,
                src_key_padding_mask=padding_mask
            )
        except RuntimeError as e:
            # Provide more information about the error
            print(f"Error in transformer: {str(e)}")
            print(f"x shape: {x.shape}, causal_mask shape: {causal_mask.shape}, padding_mask shape: {padding_mask.shape}")
            # Re-raise to allow caller to handle
            raise e
        
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
                
                # IMPORTANT FIX 3: Ensure sequence length doesn't exceed model's max_seq_length
                if seq_len > self.max_seq_length:
                    print(f"Warning: Sequence length {seq_len} exceeds max_seq_length {self.max_seq_length}. Truncating.")
                    sequences = sequences[:, :self.max_seq_length]
                    seq_len = self.max_seq_length
                
                # Safety check - ensure no out-of-range indices
                if torch.max(sequences) >= self.vocab_size:
                    print(f"Warning: Input contains indices >= vocab_size ({torch.max(sequences).item()} >= {self.vocab_size})")
                    # Clamp values to valid range
                    sequences = torch.clamp(sequences, 0, self.vocab_size - 1)
                
                # IMPORTANT FIX 4: Ensure sequences are on the correct device
                sequences = sequences.to(device)
                
                # Create target sequences (shifted input)
                targets = torch.zeros_like(sequences)
                if seq_len > 1:
                    targets[:, :-1] = sequences[:, 1:]
                
                # Forward pass through model
                output = self.model(sequences)  # [batch_size, seq_len, vocab_size]
                
                # We will only compute loss on non-padding tokens
                valid_pos_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=sequences.device)
                for b in range(batch_size):
                    # Find the last non-zero token position
                    non_zeros = (sequences[b] > 0).nonzero(as_tuple=True)[0]
                    if len(non_zeros) > 0:
                        # -1 because we don't predict after the last position
                        last_valid_pos = min(non_zeros[-1].item(), seq_len - 2)
                        valid_pos_mask[b, :last_valid_pos+1] = True
                
                # Create target mask (positions where target is not padding)
                target_mask = (targets > 0)
                
                # Final mask combines both conditions
                final_mask = valid_pos_mask & target_mask
                
                # Reshape for loss calculation
                flat_output = output.reshape(-1, self.vocab_size)
                flat_targets = targets.reshape(-1)
                flat_mask = final_mask.reshape(-1)
                
                # Skip batch if no valid positions
                if not torch.any(flat_mask):
                    print("Warning: No valid positions in batch, skipping")
                    continue
                
                # IMPORTANT FIX 5: Safer approach to masked selection
                masked_indices = torch.nonzero(flat_mask).squeeze(1)
                if masked_indices.numel() == 0:
                    print("Warning: No valid indices after masking, skipping batch")
                    continue
                    
                # Safely select the valid outputs and targets
                masked_output = flat_output[masked_indices]
                masked_targets = flat_targets[masked_indices]
                
                # Double-check targets are valid
                if torch.any(masked_targets >= self.vocab_size):
                    print(f"Warning: Target indices out of range, clamping")
                    masked_targets = torch.clamp(masked_targets, 0, self.vocab_size - 1)
                
                # Calculate loss and backprop
                loss = self.criterion(masked_output, masked_targets) / self.accumulation_steps
                loss.backward()
                    
                # Update weights after accumulation_steps
                if (i + 1) % self.accumulation_steps == 0:
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    torch.cuda.empty_cache()
                
                # Track statistics
                epoch_loss += loss.item() * self.accumulation_steps
                epoch_batches += 1
                
            except RuntimeError as e:
                print(f"Error in batch {i}: {str(e)}")
                torch.cuda.empty_cache()
                self.optimizer.zero_grad()
                continue
            
        # Handle any remaining gradients
        if epoch_batches % self.accumulation_steps != 0 and epoch_batches > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        # Calculate average loss
        avg_loss = epoch_loss / epoch_batches if epoch_batches > 0 else float('inf')
        
        # Update scheduler
        self.scheduler.step()
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
                    # Handle empty or None tensors
                    if item is None:
                        preds.append(None)
                        continue
                        
                    # Add batch dimension if needed
                    if item.dim() == 1:
                        item = item.unsqueeze(0)
                    
                    # Safety check for out-of-range indices
                    if torch.max(item) >= self.vocab_size:
                        print(f"Warning: Input contains indices >= vocab_size ({torch.max(item).item()} >= {self.vocab_size})")
                        item = torch.clamp(item, 0, self.vocab_size - 1)
                    
                    # Move to device
                    item = item.to(device)
                    
                    # Get model predictions
                    output = self.model(item)  # [batch_size, seq_len, vocab_size]
                    
                    batch_size, seq_len = item.size()
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
                            
                            # IMPORTANT FIX: Convert to CPU and make sure it's a 1D tensor of integers
                            top_indices = top_indices.cpu().tolist()
                            
                            # Add to batch predictions
                            batch_predictions.append(top_indices)
                        else:
                            # If no non-zero tokens, return prediction for first position
                            logits = output[b, 0]
                            scaled_logits = logits / temperature
                            probs = torch.softmax(scaled_logits, dim=-1)
                            _, top_indices = torch.topk(probs, min(3, self.vocab_size))
                            
                            # IMPORTANT FIX: Convert to CPU and make sure it's a 1D list of integers
                            top_indices = top_indices.cpu().tolist()
                            
                            batch_predictions.append(top_indices)
                    
                    # Add batch predictions to overall predictions
                    if batch_predictions:
                        # No need to stack here since we're returning lists of integers
                        preds.append(batch_predictions[0] if len(batch_predictions) == 1 else batch_predictions)
                    else:
                        preds.append(None)
                    
                except Exception as e:
                    print(f"Warning: Error in prediction: {str(e)}")
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