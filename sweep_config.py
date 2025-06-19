# WandB Sweep Configuration for Word2Vec Model

# Sweep configuration with appropriate hyperparameter ranges
sweep_config = {
    'method': 'bayes',  # 'grid', 'random', or 'bayes'
    'metric': {
        'name': 'best_loss',
        'goal': 'minimize'
    },
    'parameters': {
        # Learning rate - log uniform distribution for better exploration
        'learning_rate': {
            'min': 0.0001,  # 1e-4
            'max': 0.01,    # 1e-2
            'distribution': 'log_uniform'
        },
        
        # Batch size - discrete values
        'batch_size': {
            'values': [512, 1024, 2048]
        },
        
        # Embedding dimension - discrete values
        'embedding_dim': {
            'values': [64, 128, 256]
        },
        
        # Optimizer choice
        'optimizer': {
            'values': ['adam', 'adamw', 'sgd']
        },
        
        # Weight decay for regularization
        'weight_decay': {
            'min': 0.0,
            'max': 0.01,
            'distribution': 'uniform'
        },
        
        # Gradient clipping threshold
        'grad_clip': {
            'min': 0.1,
            'max': 5.0,
            'distribution': 'uniform'
        },
        
        # Learning rate scheduler
        'scheduler': {
            'values': ['none', 'cosine', 'step']
        },
        
        # Fixed parameters
        'epochs': {
            'value': 5
        },
        'seed': {
            'value': 42
        },
        'early_stopping': {
            'value': True
        },
        'patience': {
            'value': 0.1  # 10% tolerance for early stopping
        }
    }
}

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