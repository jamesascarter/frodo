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

def save_doc_embedding_to_redis(doc_id, embedding, text):
    try:
        r.hset(doc_id, mapping={
            'embedding': embedding.astype(np.float32).tobytes(),
            'text': text,
            'doc_id': doc_id
        })
    except Exception as e:
        logger.error(f"Failed to save document {doc_id}: {e}")
        return False
    return True

# Process documents in batches
batch_size = 32
total_saved = 0
total_failed = 0

# Use fast mode if --fast flag is passed
test_mode = args.fast
num_test_docs = 5 if test_mode else None

logger.info("Processing documents...")
doc_keys = ds.d_keys[:num_test_docs] if test_mode else ds.d_keys
logger.info(f"Will process {len(doc_keys)} documents {'(test mode)' if test_mode else ''}")

for doc_key in tqdm(doc_keys, desc="Saving doc embeddings"):
    try:
        doc_text = ds.docs[doc_key]
        logger.info(f"\nProcessing document: {doc_text[:100]}...")  # Show first 100 chars
        
        doc_emb = ds.to_emb(doc_text)
        
        if doc_emb is not None:
            with torch.no_grad():  # Ensure we're in inference mode
                doc_emb = two.doc(doc_emb).detach().cpu().numpy()
            if save_doc_embedding_to_redis(f"doc:{doc_key}", doc_emb, doc_text):
                total_saved += 1
                logger.info(f"Successfully saved document {doc_key}")
                logger.info(f"Embedding shape: {doc_emb.shape}")
        else:
            total_failed += 1
            logger.warning(f"Could not generate embedding for document {doc_key}")
            
    except Exception as e:
        total_failed += 1
        logger.error(f"Error processing document {doc_key}: {e}")
        continue
        
    # Clear memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

logger.info(f"\nCompleted! Saved {total_saved} documents to Redis Cloud.")
if total_failed > 0:
    logger.warning(f"Failed to process {total_failed} documents.")

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