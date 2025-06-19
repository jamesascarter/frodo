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
    for epoch in range(3):  # Reduced epochs for faster sweep
        epoch_loss = 0.0
        batch_count = 0
        
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
                
                epoch_loss += loss.item()
                batch_count += 1
                
                # Log every 1000 batches
                if idx % 1000 == 0:
                    wandb.log({'loss': loss.item(), 'epoch': epoch, 'batch': idx})
                
            except Exception as e:
                continue
        
        # Calculate average loss
        avg_loss = epoch_loss / batch_count if batch_count > 0 else float('inf')
        wandb.log({'epoch_loss': avg_loss, 'epoch': epoch})
        
        # Track best loss
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(w2v.state_dict(), f'./checkpoints/sweep_best_{wandb.run.id}.pth')
    
    wandb.log({'best_loss': best_loss})
    logger.info(f"Sweep completed. Best loss: {best_loss:.4f}")


def main():
    """Simple sweep configuration"""
    
    # Simple sweep config
    sweep_config = {
        'method': 'random',  # Simple random search
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