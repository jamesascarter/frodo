import wandb
import torch
import models
import pickle
import dataset
import datetime
import tqdm
import os
import glob
import logging

def train():
    """Training function that runs directly (like simple_sweep.py)"""
    # Initialize wandb with sweep config
    wandb.init()
    config = wandb.config
    
    # Set up logging
    ts = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    torch.manual_seed(42)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load tokenizer
    with open('./corpus/tokeniser.pkl', 'rb') as f: 
        tkns = pickle.load(f)
    words_to_ids, ids_to_words = tkns['words_to_ids'], tkns['ids_to_words']
    
    # Load w2v checkpoint
    w2v_files = glob.glob('./checkpoints/*.w2v.pth')
    if not w2v_files:
        raise FileNotFoundError("No Word2Vec checkpoints found")
    w2v_checkpoint = max(w2v_files, key=os.path.getctime)
    logger.info(f"Found latest Word2Vec checkpoint: {w2v_checkpoint}")
    
    # Load checkpoint to get embedding dimension
    checkpoint = torch.load(w2v_checkpoint, map_location=dev)
    embedding_dim = checkpoint['emb.weight'].shape[1]
    logger.info(f"Checkpoint embedding dimension: {embedding_dim}")
    
    # Load w2v model
    w2v = models.SkipGram(voc=len(words_to_ids), emb=embedding_dim).to(dev)
    w2v.load_state_dict(checkpoint)
    
    # Create dataset and dataloader
    ds = dataset.Triplets(w2v.emb, words_to_ids)
    dl = torch.utils.data.DataLoader(ds, batch_size=config.batch_size, shuffle=True)
    
    # Create Towers model
    two = models.Towers(emb=embedding_dim).to(dev)
    torch.save(two.state_dict(), f'./checkpoints/{ts}.0.0.two.pth')
    print('two:', sum(p.numel() for p in two.parameters()))
    opt = torch.optim.Adam(two.parameters(), lr=config.learning_rate)
    
    # Training loop
    for epoch in range(config.epochs):
        prgs = tqdm.tqdm(dl, desc=f"Epoch {epoch + 1}", leave=False)
        for idx, (qry, pos, neg) in enumerate(prgs):
            qry, pos, neg = qry.to(dev), pos.to(dev), neg.to(dev)
            loss = two(qry, pos, neg, mrg=config.margin)
            opt.zero_grad()
            loss.backward()
            opt.step()
            wandb.log({'loss': loss.item()})
            if idx % 50 == 0: 
                torch.save(two.state_dict(), f'./checkpoints/{ts}.{epoch}.{idx}.two.pth')

# Sweep configuration
sweep_config = {
    'method': 'random',
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