#
#
#
import torch
import models
import dataset
import datetime
import pickle
import wandb
import tqdm
import logging
import os


#
#
#
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'./logs/training_{ts}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create logs directory if it doesn't exist
os.makedirs('./logs', exist_ok=True)

torch.manual_seed(42)
ts = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')

logger.info(f"Starting training at {ts}")
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Device: {dev}")
logger.info(f"CUDA available: {torch.cuda.is_available()}")

with open('./corpus/tokeniser.pkl', 'rb') as f: tkns = pickle.load(f)
words_to_ids, ids_to_words = tkns['words_to_ids'], tkns['ids_to_words']
logger.info(f"Vocabulary size: {len(words_to_ids)}")

w2v = models.SkipGram(voc=len(words_to_ids), emb=128).to(dev)
torch.save(w2v.state_dict(), f'./checkpoints/{ts}.0.w2v.pth')
logger.info(f"Model parameters: {sum(p.numel() for p in w2v.parameters()):,}")

opt = torch.optim.Adam(w2v.parameters(), lr=0.003)
wandb.init(project='mlx6-week-02-mrc')

ds = dataset.Window('./corpus/tokens.txt')
logger.info(f"Dataset size: {len(ds)}")

dl = torch.utils.data.DataLoader(
    ds,
    batch_size=4096,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=True,
    drop_last=True
)
logger.info(f"DataLoader configured with batch_size=4096, num_workers=8")

# Training loop with comprehensive logging
for epoch in range(5):
    logger.info(f"Starting epoch {epoch + 1}/5")
    epoch_loss = 0.0
    batch_count = 0
    
    prgs = tqdm.tqdm(dl, desc=f"Epoch {epoch + 1}", leave=False)
    for idx, (inpt, trgs) in enumerate(prgs):
        try:
            # Log batch info every 1000 batches
            if idx % 1000 == 0:
                logger.info(f"Epoch {epoch + 1}, Batch {idx}: Processing batch of size {inpt.size()}")
            
            inpt, trgs = inpt.to(dev, non_blocking=True), trgs.to(dev, non_blocking=True)
            rand = torch.randint(0, len(words_to_ids), (inpt.size(0), 2), device=dev)
            
            opt.zero_grad()
            loss = w2v(inpt, trgs, rand)
            loss.backward()
            opt.step()
            
            # Update metrics
            epoch_loss += loss.item()
            batch_count += 1
            
            # Log detailed info every 1000 batches
            if idx % 1000 == 0:
                avg_loss = epoch_loss / batch_count
                logger.info(f"Epoch {epoch + 1}, Batch {idx}: Loss={loss.item():.4f}, Avg Loss={avg_loss:.4f}")
                logger.info(f"GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB allocated, {torch.cuda.memory_reserved()/1024**3:.2f}GB reserved")
            
            wandb.log({'loss': loss.item()})
            if idx % 10_000 == 0: 
                torch.save(w2v.state_dict(), f'./checkpoints/{ts}.{epoch}.{idx}.w2v.pth')
                logger.info(f"Checkpoint saved: {ts}.{epoch}.{idx}.w2v.pth")
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.error(f"GPU OOM at epoch {epoch + 1}, batch {idx}. Trying to recover...")
                torch.cuda.empty_cache()
                continue
            else:
                logger.error(f"Runtime error at epoch {epoch + 1}, batch {idx}: {str(e)}")
                raise e
        except Exception as e:
            logger.error(f"Unexpected error at epoch {epoch + 1}, batch {idx}: {str(e)}")
            raise e
    
    # Log epoch summary
    avg_epoch_loss = epoch_loss / batch_count
    logger.info(f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")
    logger.info(f"Total batches processed: {batch_count}")

logger.info("Training completed successfully")
wandb.finish()
