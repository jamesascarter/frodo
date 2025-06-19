# WandB Sweep Configuration for Word2Vec Model

import wandb

# Define the sweep configuration
sweep_config = {
    'method': 'random',  # or 'grid', 'bayes'
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
            'distribution': 'log_uniform'
        },
        'margin': {
            'min': 0.1,
            'max': 0.5,
            'distribution': 'uniform'
        },
        'embedding_dim': {
            'values': [64, 128, 256]
        }
    }
}

# Initialize the sweep
sweep_id = wandb.sweep(sweep_config, project='mlx6-week-02-two')

print(f"Sweep ID: {sweep_id}")

# Conditional parameters for step scheduler
sweep_config['parameters']['step_size'] = {
    'values': [2, 3],
    'parent': 'scheduler',
    'parent_values': ['step']
}

sweep_config['parameters']['gamma'] = {
    'min': 0.1,
    'max': 0.9,
    'distribution': 'uniform',
    'parent': 'scheduler',
    'parent_values': ['step']
}

# Conditional parameters for SGD optimizer
sweep_config['parameters']['momentum'] = {
    'min': 0.8,
    'max': 0.99,
    'distribution': 'uniform',
    'parent': 'optimizer',
    'parent_values': ['sgd']
}

# Alternative configurations for different scenarios

# Quick sweep (fewer trials, smaller ranges)
quick_sweep_config = {
    'method': 'random',
    'metric': {
        'name': 'best_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'learning_rate': {
            'values': [0.001, 0.005, 0.01]
        },
        'batch_size': {
            'values': [1024, 2048]
        },
        'embedding_dim': {
            'values': [128, 256]
        },
        'optimizer': {
            'values': ['adam', 'adamw']
        },
        'weight_decay': {
            'values': [0.0, 0.001, 0.01]
        },
        'grad_clip': {
            'values': [1.0, 2.0, 5.0]
        },
        'scheduler': {
            'values': ['none', 'cosine']
        },
        'epochs': {
            'value': 3
        },
        'seed': {
            'value': 42
        },
        'early_stopping': {
            'value': True
        },
        'patience': {
            'value': 0.1
        }
    }
}

# Grid search configuration (exhaustive search)
grid_sweep_config = {
    'method': 'grid',
    'metric': {
        'name': 'best_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'learning_rate': {
            'values': [0.001, 0.005]
        },
        'batch_size': {
            'values': [1024, 2048]
        },
        'embedding_dim': {
            'values': [128, 256]
        },
        'optimizer': {
            'values': ['adam', 'adamw']
        },
        'weight_decay': {
            'values': [0.0, 0.001]
        },
        'grad_clip': {
            'values': [1.0, 2.0]
        },
        'scheduler': {
            'values': ['none', 'cosine']
        },
        'epochs': {
            'value': 3
        },
        'seed': {
            'value': 42
        },
        'early_stopping': {
            'value': True
        },
        'patience': {
            'value': 0.1
        }
    }
} 