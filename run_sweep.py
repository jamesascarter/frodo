import wandb
import subprocess
import sys
import argparse

def train():
    # This function will be called by wandb agent
    with wandb.init() as run:
        # Get hyperparameters from wandb
        config = wandb.config
        
        # Build command with hyperparameters
        cmd = [
            sys.executable, '04_train.two.py',
            '--epochs', str(config.epochs),
            '--batch_size', str(config.batch_size),
            '--learning_rate', str(config.learning_rate),
            '--margin', str(config.margin),
            '--embedding_dim', str(config.embedding_dim)
        ]
        
        # Run the training script
        subprocess.run(cmd)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run wandb sweep')
    parser.add_argument('--sweep_id', type=str, required=True, help='Wandb sweep ID')
    parser.add_argument('--count', type=int, default=10, help='Number of trials to run (default: 10)')
    args = parser.parse_args()
    
    # Run the agent with provided sweep ID
    wandb.agent(args.sweep_id, function=train, count=args.count) 