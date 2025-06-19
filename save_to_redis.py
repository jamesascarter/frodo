import torch
import torch.nn as nn
import pickle
import json
import redis
import numpy as np
from tqdm import tqdm
import os
import glob
from dotenv import load_dotenv
import logging
import models
import dataset
import argparse
import psutil
import gc

# Set up argument parser
parser = argparse.ArgumentParser(description='Save document embeddings to Redis')
parser.add_argument('--fast', action='store_true', help='Process only 5 documents (test mode)')
args = parser.parse_args()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

# Redis Cloud connection
REDIS_HOST = os.environ.get('REDIS_HOST')
REDIS_PORT = int(os.environ.get('REDIS_PORT'))
REDIS_PASSWORD = os.environ.get('REDIS_PASSWORD')

if not all([REDIS_HOST, REDIS_PORT, REDIS_PASSWORD]):
    raise ValueError("Missing Redis credentials. Please check your .env file.")

logger.info("Connecting to Redis Cloud...")
r = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password=REDIS_PASSWORD,
    decode_responses=False  # binary-safe
)

try:
    r.ping()
    logger.info("Successfully connected to Redis Cloud")
except redis.ConnectionError as e:
    logger.error(f"Failed to connect to Redis: {e}")
    raise

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

# Load tokenizer
logger.info("Loading tokenizer...")
with open('./corpus/tokeniser.pkl', 'rb') as f: 
    tkns = pickle.load(f)
words_to_ids, ids_to_words = tkns['words_to_ids'], tkns['ids_to_words']
logger.info(f"Loaded tokenizer with {len(words_to_ids)} tokens")

def get_latest_checkpoint(pattern, model_type):
    """Find the latest checkpoint file matching the pattern and model type."""
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No checkpoint files found matching pattern: {pattern}")
    
    # Filter files by model type
    files = [f for f in files if model_type in f]
    if not files:
        raise FileNotFoundError(f"No {model_type} checkpoint files found")
    
    return max(files, key=os.path.getctime)

# Load Word2Vec model
emb_size = 128  # Match the Two Towers checkpoint
logger.info("Loading Word2Vec model...")
w2v = models.SkipGram(voc=len(words_to_ids), emb=emb_size).to(dev)

# Find latest Word2Vec checkpoint
w2v_checkpoint = get_latest_checkpoint('./checkpoints/*.w2v.pth', 'w2v')
logger.info(f"Found latest Word2Vec checkpoint: {w2v_checkpoint}")

try:
    checkpoint = torch.load(w2v_checkpoint, map_location=dev)
    w2v.load_state_dict(checkpoint)
    w2v.eval()
    logger.info(f"Successfully loaded Word2Vec checkpoint")
except Exception as e:
    logger.error(f"Failed to load Word2Vec checkpoint: {e}")
    raise

# Load Two Towers model
logger.info("Loading Two Towers model...")
two = models.Towers(emb=emb_size).to(dev)

# Find latest Two Towers checkpoint
two_checkpoint = get_latest_checkpoint('./checkpoints/*.two.pth', 'two')
logger.info(f"Found latest Two Towers checkpoint: {two_checkpoint}")

try:
    checkpoint = torch.load(two_checkpoint, map_location=dev)
    # Check if it's a Word2Vec checkpoint
    if 'emb.weight' in checkpoint:
        raise ValueError("This appears to be a Word2Vec checkpoint, not a Two Towers checkpoint")
    two.load_state_dict(checkpoint)
    two.eval()
    logger.info(f"Successfully loaded Two Towers checkpoint")
except Exception as e:
    logger.error(f"Failed to load Two Towers checkpoint: {e}")
    raise

# Initialize dataset
logger.info("Loading dataset...")
ds = dataset.Triplets(w2v.emb, words_to_ids)
logger.info(f"Found {len(ds.d_keys)} documents")

# Ensure RediSearch index exists
try:
    r.execute_command(f"FT.INFO doc_index")
    logger.info(f"Index 'doc_index' exists")
except redis.ResponseError:
    logger.info(f"Creating index 'doc_index'...")
    try:
        # Create index with L2 distance and better HNSW parameters
        r.execute_command(
            "FT.CREATE doc_index ON HASH PREFIX 1 doc: "
            f"SCHEMA embedding VECTOR HNSW 12 TYPE FLOAT32 "
            f"DIM {emb_size} DISTANCE_METRIC L2 "
            "text TEXT doc_id TAG"
        )
        logger.info(f"Index 'doc_index' created successfully")
    except redis.ResponseError as e:
        logger.error(f"Failed to create index: {e}")
        raise

def save_batch_to_redis(doc_ids, embeddings, texts):
    """Save a batch of documents to Redis in a pipeline."""
    try:
        pipe = r.pipeline(transaction=False)
        for doc_id, embedding, text in zip(doc_ids, embeddings, texts):
            pipe.hset(f"doc:{doc_id}", mapping={
                'embedding': embedding.astype(np.float32).tobytes(),
                'text': text,
                'doc_id': f"doc:{doc_id}"
            })
        pipe.execute()
        return True
    except Exception as e:
        logger.error(f"Failed to save batch: {e}")
        return False

def log_memory_usage():
    """Log current memory usage"""
    process = psutil.Process()
    mem_info = process.memory_info()
    logger.info(f"Memory usage: {mem_info.rss / 1024 / 1024:.2f} MB")

# Process documents in larger batches
batch_size = 512  # Increased from 128 for better throughput
max_retries = 3   # Add retries for batch processing
total_saved = 0
total_failed = 0

# Use fast mode if --fast flag is passed
test_mode = args.fast
num_test_docs = 5 if test_mode else None

logger.info("Processing documents...")
doc_keys = ds.d_keys[:num_test_docs] if test_mode else ds.d_keys
logger.info(f"Will process {len(doc_keys)} documents {'(test mode)' if test_mode else ''}")
log_memory_usage()

# Process in batches
for i in tqdm(range(0, len(doc_keys), batch_size), desc="Processing batches"):
    retry_count = 0
    while retry_count < max_retries:
        try:
            batch_keys = doc_keys[i:i + batch_size]
            batch_texts = [ds.docs[key] for key in batch_keys]
            batch_embs = []
            valid_indices = []
            valid_keys = []
            valid_texts = []

            # Process each document in the batch
            for idx, (doc_key, doc_text) in enumerate(zip(batch_keys, batch_texts)):
                try:
                    doc_emb = ds.to_emb(doc_text)
                    if doc_emb is not None:
                        batch_embs.append(doc_emb)
                        valid_indices.append(idx)
                        valid_keys.append(doc_key)
                        valid_texts.append(doc_text)
                    else:
                        total_failed += 1
                except Exception as e:
                    total_failed += 1
                    continue

            if batch_embs:
                # Convert list of embeddings to tensor
                batch_embs = torch.stack(batch_embs).to(dev)
                
                with torch.no_grad():  # Ensure we're in inference mode
                    # Process entire batch through the model
                    batch_embs = two.doc(batch_embs)
                    # Normalize the embeddings
                    batch_embs = torch.nn.functional.normalize(batch_embs, p=2, dim=1)
                    batch_embs = batch_embs.detach().cpu().numpy()

                # Save batch to Redis
                if save_batch_to_redis(valid_keys, batch_embs, valid_texts):
                    total_saved += len(valid_keys)
                    if i % 10 == 0:  # Log every 10 batches
                        logger.info(f"Successfully saved batch of {len(valid_keys)} documents")
                        logger.info(f"Total saved so far: {total_saved}")
                        log_memory_usage()
            
            # Clear memory
            del batch_embs, valid_indices, valid_keys, valid_texts
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            break  # Success - exit retry loop
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                retry_count += 1
                logger.warning(f"Out of memory error, retry {retry_count}/{max_retries}")
                # Clear everything possible
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                if retry_count == max_retries:
                    logger.error("Failed after max retries, reducing batch size")
                    batch_size = batch_size // 2
                    retry_count = 0
            else:
                raise e

logger.info(f"\nProcessing complete!")
logger.info(f"Total documents saved: {total_saved}")
logger.info(f"Total documents failed: {total_failed}")
log_memory_usage()

# Verify the saved documents
logger.info("\nVerifying saved documents...")
for doc_key in doc_keys:
    redis_key = f"doc:{doc_key}"
    try:
        saved_data = r.hgetall(redis_key)
        if saved_data:
            emb = np.frombuffer(saved_data[b'embedding'], dtype=np.float32)
            text = saved_data[b'text'].decode('utf-8')
            logger.info(f"\nDocument {doc_key}:")
            logger.info(f"Text: {text[:100]}...")  # Show first 100 chars
            logger.info(f"Embedding shape: {emb.shape}")
        else:
            logger.warning(f"Document {doc_key} not found in Redis")
    except Exception as e:
        logger.error(f"Error verifying document {doc_key}: {e}")