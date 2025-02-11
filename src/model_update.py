import torch
from modules.transformer_predictor import TransformerPredictor

# Load the model
work_dir = "work"  # Replace with your model path
predictor = TransformerPredictor.load(work_dir)

# Print current learning rate
print(f"Current learning rate: {predictor.optimizer.param_groups[0]['lr']}")

# Get optimizer state dict
optimizer_state = predictor.optimizer.state_dict()

# Update learning rate only in existing fields
new_lr = 0.001  # New learning rate
for group in optimizer_state['param_groups']:
    if 'lr' in group:
        group['lr'] = new_lr
    if 'initial_lr' in group:
        group['initial_lr'] = new_lr

# Load modified optimizer state back
predictor.optimizer.load_state_dict(optimizer_state)

# Get scheduler state dict
scheduler_state = predictor.scheduler.state_dict()
if 'base_lrs' in scheduler_state:
    scheduler_state['base_lrs'] = [new_lr] * len(scheduler_state['base_lrs'])

# Load modified scheduler state back
predictor.scheduler.load_state_dict(scheduler_state)

# Verify the change
current_lr = predictor.optimizer.param_groups[0]['lr']
print(f"Updated learning rate: {current_lr}")

assert current_lr == new_lr, "Learning rate update failed"

# Save the updated model
predictor.save(work_dir)