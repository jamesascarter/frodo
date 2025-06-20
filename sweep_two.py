import wandb
import subprocess
import sys

def train():
    """Function to be called by wandb agent"""
    with wandb.init() as run:
        # Get hyperparameters from wandb
        config = wandb.config
        
        # Build command with hyperparameters
        cmd = [
            sys.executable, '04_train.two.py',
            '--epochs', str(config.epochs),
            '--batch_size', str(config.batch_size),
            '--learning_rate', str(config.learning_rate),
            '--margin', str(config.margin)
        ]
        
        # Run the training script
        subprocess.run(cmd)

# Sweep configuration
sweep_config = {
    'method': 'bayes',
    'name': 'towers-hyperparameter-sweep',
    'metric': {
        'name': 'loss',
        'goal': 'minimize'
    },
    'parameters': {
        'epochs': {
            'values': [1, 2, 3]
        },
        'batch_size': {
            'values': [128, 256, 512]
        },
        'learning_rate': {
            'min': 0.0001,
            'max': 0.01,
            'distribution': 'log_uniform_values'
        },
        'margin': {
            'min': 0.1,
            'max': 0.5,
            'distribution': 'uniform'
        }
    }
}

# Initialize wandb
wandb.login()

# Create and run sweep
sweep_id = wandb.sweep(sweep_config, project='mlx6-week-02-two')
print(f"Sweep created with ID: {sweep_id}")

# Run 10 trials
print("Starting sweep with 10 trials...")
wandb.agent(sweep_id, function=train, count=10)
print("Sweep completed!") 