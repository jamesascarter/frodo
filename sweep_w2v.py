import wandb
import torch
import models as models
import dataset
import datetime
import pickle
import tqdm
import logging
import os
import math


def train_sweep():
    """Training function for wandb sweep"""
    
    # Initialize wandb with sweep config
    wandb.init()
    
    # Get hyperparameters from sweep
    config = wandb.config
    
    # Create logs directory if it doesn't exist
    os.makedirs('./logs', exist_ok=True)
    
    # Set up logging
    ts = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'./logs/sweep_training_{ts}.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    torch.manual_seed(config.seed)
    
    logger.info(f"Starting sweep training with config: {config}")
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {dev}")
    
    # Load tokenizer
    with open('./corpus/tokeniser.pkl', 'rb') as f: 
        tkns = pickle.load(f)
    words_to_ids, ids_to_words = tkns['words_to_ids'], tkns['ids_to_words']
    logger.info(f"Vocabulary size: {len(words_to_ids)}")
    
    # Create model with sweep hyperparameters
    w2v = models.SkipGram(voc=len(words_to_ids), emb=config.embedding_dim).to(dev)
    logger.info(f"Model parameters: {sum(p.numel() for p in w2v.parameters()):,}")
    
    # Optimizer with sweep learning rate
    if config.optimizer == 'adam':
        opt = torch.optim.Adam(w2v.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optimizer == 'sgd':
        opt = torch.optim.SGD(w2v.parameters(), lr=config.learning_rate, momentum=config.momentum, weight_decay=config.weight_decay)
    else:
        opt = torch.optim.AdamW(w2v.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    # Learning rate scheduler
    if config.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=config.epochs)
    elif config.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=config.step_size, gamma=config.gamma)
    else:
        scheduler = None
    
    # Dataset and dataloader
    ds = dataset.Window('./corpus/tokens.txt')
    logger.info(f"Dataset size: {len(ds)}")
    
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=config.batch_size,
        shuffle=True
    )
    logger.info(f"DataLoader configured with batch_size={config.batch_size}")
    
    # Training loop
    best_loss = float('inf')
    for epoch in range(config.epochs):
        logger.info(f"Starting epoch {epoch + 1}/{config.epochs}")
        epoch_loss = 0.0
        batch_count = 0
        
        prgs = tqdm.tqdm(dl, desc=f"Epoch {epoch + 1}", leave=False)
        for idx, (inpt, trgs) in enumerate(prgs):
            try:
                inpt, trgs = inpt.to(dev, non_blocking=True), trgs.to(dev, non_blocking=True)
                rand = torch.randint(0, len(words_to_ids), (inpt.size(0), 2), device=dev)
                
                opt.zero_grad()
                loss = w2v(inpt, trgs, rand)
                
                # Check for NaN or infinite loss
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.error(f"NaN/Inf loss detected at epoch {epoch + 1}, batch {idx}. Loss: {loss.item()}")
                    continue
                
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(w2v.parameters(), max_norm=config.grad_clip)
                
                # Check for NaN gradients
                has_nan_grad = False
                for param in w2v.parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        has_nan_grad = True
                        break
                
                if has_nan_grad:
                    logger.error(f"NaN gradients detected at epoch {epoch + 1}, batch {idx}. Skipping update.")
                    continue
                
                opt.step()
                
                # Update metrics
                epoch_loss += loss.item()
                batch_count += 1
                
                # Log to wandb
                wandb.log({
                    'loss': loss.item(),
                    'epoch': epoch,
                    'batch': idx,
                    'learning_rate': opt.param_groups[0]['lr']
                })
                
            except Exception as e:
                logger.error(f"Error at epoch {epoch + 1}, batch {idx}: {str(e)}")
                continue
        
        # Update scheduler
        if scheduler:
            scheduler.step()
        
        # Calculate average loss for epoch
        avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else float('inf')
        logger.info(f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")
        
        # Log epoch metrics
        wandb.log({
            'epoch_loss': avg_epoch_loss,
            'epoch': epoch
        })
        
        # Track best loss
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            # Save best model
            torch.save(w2v.state_dict(), f'./checkpoints/sweep_best_{wandb.run.id}.pth')
        
        # Early stopping
        if config.early_stopping and epoch > 0:
            if avg_epoch_loss > best_loss * (1 + config.patience):
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break
    
    # Log final metrics
    wandb.log({
        'best_loss': best_loss,
        'final_loss': avg_epoch_loss
    })
    
    logger.info(f"Sweep training completed. Best loss: {best_loss:.4f}")


def main():
    """Main function to set up and run the sweep"""
    
    # Sweep configuration
    sweep_config = {
        'method': 'bayes',  # 'grid', 'random', or 'bayes'
        'metric': {
            'name': 'best_loss',
            'goal': 'minimize'
        },
        'parameters': {
            'learning_rate': {
                'min': 0.0001,
                'max': 0.01,
                'distribution': 'log_uniform'
            },
            'batch_size': {
                'values': [512, 1024, 2048]
            },
            'embedding_dim': {
                'values': [64, 128, 256]
            },
            'optimizer': {
                'values': ['adam', 'adamw', 'sgd']
            },
            'weight_decay': {
                'min': 0.0,
                'max': 0.01,
                'distribution': 'uniform'
            },
            'grad_clip': {
                'min': 0.1,
                'max': 5.0,
                'distribution': 'uniform'
            },
            'scheduler': {
                'values': ['none', 'cosine', 'step']
            },
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
                'value': 0.1
            }
        }
    }
    
    # Conditional parameters (only used if scheduler is 'step')
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
    
    # Conditional parameters (only used if optimizer is 'sgd')
    sweep_config['parameters']['momentum'] = {
        'min': 0.8,
        'max': 0.99,
        'distribution': 'uniform',
        'parent': 'optimizer',
        'parent_values': ['sgd']
    }
    
    # Initialize wandb
    wandb.login()
    
    # Create sweep
    sweep_id = wandb.sweep(sweep_config, project="word2vec-sweep")
    
    print(f"Sweep created with ID: {sweep_id}")
    print("Starting sweep...")
    
    # Run the sweep
    wandb.agent(sweep_id, function=train_sweep, count=20)  # Run 20 trials
    
    print("Sweep completed!")


if __name__ == "__main__":
    main() 