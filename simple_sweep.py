import wandb
import torch
import models as models
import dataset
import datetime
import pickle
import tqdm
import logging
import os


def train_sweep():
    """Simple training function for wandb sweep"""
    
    # Initialize wandb with sweep config
    wandb.init()
    config = wandb.config
    
    # Set up logging
    ts = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'./logs/sweep_{ts}.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    torch.manual_seed(42)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load tokenizer
    with open('./corpus/tokeniser.pkl', 'rb') as f: 
        tkns = pickle.load(f)
    words_to_ids, ids_to_words = tkns['words_to_ids'], tkns['ids_to_words']
    
    # Create model
    w2v = models.SkipGram(voc=len(words_to_ids), emb=config.embedding_dim).to(dev)
    
    # Optimizer
    opt = torch.optim.Adam(w2v.parameters(), lr=config.learning_rate)
    
    # Dataset and dataloader
    ds = dataset.Window('./corpus/tokens.txt')
    dl = torch.utils.data.DataLoader(ds, batch_size=config.batch_size, shuffle=True)
    
    # Training loop
    best_loss = float('inf')
    total_steps = 0
    target_steps = 20000
    running_loss = 0.0
    
    while total_steps < target_steps:
        for idx, (inpt, trgs) in enumerate(dl):
            try:
                inpt, trgs = inpt.to(dev), trgs.to(dev)
                rand = torch.randint(0, len(words_to_ids), (inpt.size(0), 2), device=dev)
                
                opt.zero_grad()
                loss = w2v(inpt, trgs, rand)
                
                # Check for NaN
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(w2v.parameters(), max_norm=1.0)
                opt.step()
                
                total_steps += 1
                running_loss += loss.item()
                
                # Log every 1000 steps
                if total_steps % 1000 == 0:
                    avg_loss = running_loss / 1000
                    wandb.log({
                        'loss': loss.item(), 
                        'avg_loss': avg_loss,
                        'steps': total_steps,
                        'learning_rate': opt.param_groups[0]['lr']
                    })
                    
                    # Track best loss
                    if avg_loss < best_loss:
                        best_loss = avg_loss
                        torch.save(w2v.state_dict(), f'./checkpoints/sweep_best_{wandb.run.id}.pth')
                    
                    running_loss = 0.0
                
                # Check if we've reached target steps
                if total_steps >= target_steps:
                    break
                
            except Exception as e:
                continue
        
        # If we've gone through the whole dataset, break
        if total_steps >= target_steps:
            break
    
    # Log final metrics
    wandb.log({'best_loss': best_loss, 'total_steps': total_steps})
    logger.info(f"Sweep completed. Total steps: {total_steps}, Best loss: {best_loss:.4f}")


def main():
    """Simple sweep configuration"""
    
    # Simple sweep config
    sweep_config = {
        'method': 'bayes',  # Simple random search
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
            }
        }
    }
    
    # Initialize wandb
    wandb.login()
    
    # Create and run sweep
    sweep_id = wandb.sweep(sweep_config, project="word2vec-simple-sweep")
    print(f"Sweep created with ID: {sweep_id}")
    
    # Run 10 trials
    wandb.agent(sweep_id, function=train_sweep, count=10)
    print("Sweep completed!")


if __name__ == "__main__":
    main() 