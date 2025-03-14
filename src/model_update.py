import torch
from modules.transformer_predictor import TransformerPredictor
from modules.rnn_predictor import RNNPredictor
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('mode', choices=('prepare', 'process', 'train', 'test', 'tui'), help='what to run')
parser.add_argument('--work_dir', help='where to save', default='work')
parser.add_argument('--model', choices=('rnn', 'transformer'), help='what model to use', default='transformer')
parser.add_argument('--learning_rate', help='If we should reset the LR', default=None)
args = parser.parse_args()
    
# Load the model
work_dir = "work"  # Replace with your model path

if args.model == 'transformer':
    predictor = TransformerPredictor.load(args.work_dir)
elif args.model == 'rnn':
    predictor = RNNPredictor.load(args.work_dir)

# Print current learning rate
if args.learning_rate:
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