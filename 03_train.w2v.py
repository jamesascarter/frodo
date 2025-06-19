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
import argparse
import math


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--fast', type=int, choices=range(0, 101), metavar="[0-100]",
                    help='Fast mode factor (0=full training, 100=fastest possible)')
args = parser.parse_args()

# Create timestamp first
torch.manual_seed(42)
ts = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')

# Create logs directory if it doesn't exist
os.makedirs('./logs', exist_ok=True)

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

# Configure training parameters based on fast factor
fast_factor = args.fast if args.fast is not None else 0

def scale_param(min_val, max_val, fast_factor, log_scale=False):
    """Scale a parameter based on fast factor. 
    At factor 0, returns max_val. At factor 100, returns min_val."""
    if fast_factor == 0:
        return max_val
    if log_scale:
        # Logarithmic scaling
        factor = 1 - (math.log(fast_factor + 1) / math.log(101))
    else:
        # Linear scaling
        factor = 1 - (fast_factor / 100)
    return int(min_val + (max_val - min_val) * factor)

if fast_factor > 0:
    # Scale parameters based on fast factor
    batch_size = scale_param(32, 1024, fast_factor)
    epochs = scale_param(1, 5, fast_factor)
    emb_size = scale_param(16, 128, fast_factor)
    window_size = scale_param(3, 5, fast_factor)
    log_freq = scale_param(10, 1000, fast_factor, log_scale=True)
    use_wandb = False
    logger.info(f"Running in fast mode with factor {fast_factor}")
    logger.info(f"Scaled parameters:")
    logger.info(f"  - Batch size: {batch_size}")
    logger.info(f"  - Epochs: {epochs}")
    logger.info(f"  - Embedding size: {emb_size}")
    logger.info(f"  - Window size: {window_size}")
    logger.info(f"  - Log frequency: {log_freq}")
else:
    # Full training mode
    batch_size = 1024
    epochs = 5
    emb_size = 128
    window_size = 5
    log_freq = 1000
    use_wandb = True
    logger.info("Running in full training mode")

# Device selection with MPS support
if torch.backends.mps.is_available():
    dev = torch.device('mps')
    logger.info("Using MPS (Metal Performance Shaders) device")
elif torch.cuda.is_available():
    dev = torch.device('cuda')
    logger.info("Using CUDA device")
else:
    dev = torch.device('cpu')
    logger.info("Using CPU device")

logger.info(f"CUDA available: {torch.cuda.is_available()}")
if torch.backends.mps.is_available():
    logger.info(f"MPS available: {torch.backends.mps.is_available()}")

with open('./corpus/tokeniser.pkl', 'rb') as f: tkns = pickle.load(f)
words_to_ids, ids_to_words = tkns['words_to_ids'], tkns['ids_to_words']
logger.info(f"Vocabulary size: {len(words_to_ids)}")

logger.info("Loading dataset...")
ds = dataset.Window('./corpus/tokens.txt', size=window_size, fast_factor=fast_factor)
logger.info(f"Dataset size: {len(ds):,} samples")

if fast_factor > 0:
    logger.info(f"Fast mode enabled - using {100 - fast_factor}% of data")

logger.info("First few samples:")
for i in range(min(3, len(ds))):
    inp, trg = ds[i]
    logger.info(f"Sample {i}: input={inp.item()}, targets={trg.tolist()}")

logger.info("\nConfiguring DataLoader...")
dl = torch.utils.data.DataLoader(
    ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0  # Force single process loading for debugging
)
num_batches = len(dl)
logger.info(f"DataLoader configured with:")
logger.info(f"  - batch_size: {batch_size}")
logger.info(f"  - total_batches: {num_batches:,}")
logger.info(f"  - total_samples: {len(ds):,}")
logger.info(f"  - epochs: {epochs}")
logger.info(f"  - embedding_size: {emb_size}")
logger.info(f"  - window_size: {window_size}")
logger.info(f"Expected samples per epoch: {num_batches * batch_size:,}")

w2v = models.SkipGram(voc=len(words_to_ids), emb=emb_size).to(dev)
torch.save(w2v.state_dict(), f'./checkpoints/{ts}.0.w2v.pth')
logger.info(f"Model parameters: {sum(p.numel() for p in w2v.parameters()):,}")

opt = torch.optim.Adam(w2v.parameters(), lr=0.003)

if use_wandb:
    try:
        wandb.init(project='mlx6-week-02-mrc')
        logger.info("Successfully initialized wandb")
    except Exception as e:
        logger.warning(f"Failed to initialize wandb: {e}")
        logger.info("Continuing without wandb logging")
        use_wandb = False
    batch_size=1024
)
logger.info(f"DataLoader configured with batch_size=1024, num_workers=1")

# Training loop with comprehensive logging
logger.info("\nStarting training loop...")
for epoch in range(epochs):
    logger.info(f"\nEpoch {epoch + 1}/{epochs}")
    epoch_loss = 0.0
    batch_count = 0
    
    # Force progress bar to display immediately
    prgs = tqdm.tqdm(dl, 
                     desc=f"Epoch {epoch + 1}", 
                     leave=True,
                     position=0,
                     ncols=100,
                     unit='batch',
                     mininterval=0,  # Update as fast as possible
                     maxinterval=0.1)  # Maximum update interval
    
    logger.info(f"Processing {len(dl):,} batches...")
    
    for idx, (inpt, trgs) in enumerate(prgs):
        if idx == 0:
            logger.info(f"First batch shapes - input: {inpt.shape}, targets: {trgs.shape}")
        
        try:
            # Log more frequently during local testing
            log_freq = 50 if fast_factor > 0 else 1000  # Even more frequent updates
            
            # Log batch info more frequently
            if idx % log_freq == 0:
                logger.info(f"Epoch {epoch + 1}, Batch {idx}/{num_batches}: Processing batch of size {inpt.size(0)}")
            
            inpt, trgs = inpt.to(dev, non_blocking=True), trgs.to(dev, non_blocking=True)
            rand = torch.randint(0, len(words_to_ids), (inpt.size(0), 2), device=dev)
            
            opt.zero_grad()
            loss = w2v(inpt, trgs, rand)
            loss.backward()
            opt.step()
            
            # Update metrics
            current_loss = loss.item()
            epoch_loss += current_loss
            batch_count += 1
            
            # Update progress bar with current loss
            prgs.set_postfix({
                'loss': f'{current_loss:.4f}',
                'avg_loss': f'{(epoch_loss/batch_count):.4f}',
                'batch': f'{idx}/{num_batches}'
            }, refresh=True)
            
            # Log detailed info more frequently
            if idx % log_freq == 0:
                avg_loss = epoch_loss / batch_count
                logger.info(f"Epoch {epoch + 1}, Batch {idx}/{num_batches}")
                logger.info(f"  - Current Loss: {current_loss:.4f}")
                logger.info(f"  - Average Loss: {avg_loss:.4f}")
                logger.info(f"  - Progress: {(idx/num_batches)*100:.1f}%")
                
                if dev.type == 'cuda':
                    logger.info(f"  - GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB allocated")
                elif dev.type == 'mps':
                    logger.info("  - Using MPS device")
            
            if use_wandb:
                wandb.log({
                    'loss': current_loss,
                    'avg_loss': epoch_loss / batch_count,
                    'epoch': epoch + 1,
                    'batch': idx
                })
            
            # Save checkpoints less frequently in fast mode
            if fast_factor > 0 and idx % 10_000 == 0: 
                checkpoint_path = f'./checkpoints/{ts}.{epoch}.{idx}.w2v.pth'
                torch.save(w2v.state_dict(), checkpoint_path)
                logger.info(f"Checkpoint saved: {checkpoint_path}")
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.error(f"GPU OOM at epoch {epoch + 1}, batch {idx}. Trying to recover...")
                if dev.type == 'cuda':
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
    logger.info(f"\nEpoch {epoch + 1} Summary:")
    logger.info(f"  - Average Loss: {avg_epoch_loss:.4f}")
    logger.info(f"  - Total Batches: {batch_count}")
    logger.info(f"  - Samples Processed: {batch_count * batch_size:,}")

# Save final model in fast mode
if fast_factor > 0:
    final_path = f'./checkpoints/{ts}.final.w2v.pth'
    torch.save(w2v.state_dict(), final_path)
    logger.info(f"\nSaved final model: {final_path}")

logger.info("\nTraining completed successfully!")
if use_wandb:
    wandb.finish()
