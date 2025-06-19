import os
import pickle
import torch
import numpy as np
import redis
from dotenv import load_dotenv
import logging
import models
import dataset
import argparse
import glob

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set up argument parser
parser = argparse.ArgumentParser(description='Query Redis ANN for similar documents')
parser.add_argument('--top-k', type=int, default=5, help='Number of results to return')
args = parser.parse_args()

# Load environment variables
load_dotenv()

# Redis configuration
REDIS_HOST = os.environ.get('REDIS_HOST')
REDIS_PORT = int(os.environ.get('REDIS_PORT'))
REDIS_PASSWORD = os.environ.get('REDIS_PASSWORD')
INDEX_NAME = 'doc_index'
EMBEDDING_DIM = 128

if not all([REDIS_HOST, REDIS_PORT, REDIS_PASSWORD]):
    raise ValueError("Missing Redis credentials. Please check your .env file.")

# Device selection
if torch.backends.mps.is_available():
    dev = torch.device('mps')
    logger.info("Using MPS (Metal Performance Shaders) device")
elif torch.cuda.is_available():
    dev = torch.device('cuda')
    logger.info("Using CUDA device")
else:
    dev = torch.device('cpu')
    logger.info("Using CPU device")

# Connect to Redis
logger.info("Connecting to Redis Cloud...")
r = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    password=REDIS_PASSWORD,
    decode_responses=False
)

try:
    r.ping()
    logger.info("Successfully connected to Redis Cloud")
except redis.ConnectionError as e:
    logger.error(f"Failed to connect to Redis: {e}")
    raise

# Ensure RediSearch index exists
try:
    r.execute_command(f"FT.INFO {INDEX_NAME}")
    logger.info(f"Index '{INDEX_NAME}' exists")
except redis.ResponseError:
    logger.info(f"Creating index '{INDEX_NAME}'...")
    try:
        r.execute_command(
            f"FT.CREATE {INDEX_NAME} ON HASH PREFIX 1 doc: "
            f"SCHEMA embedding VECTOR HNSW 12 TYPE FLOAT32 "
            f"DIM {EMBEDDING_DIM} DISTANCE_METRIC L2 "
            "text TEXT doc_id TAG"
        )
        logger.info(f"Index '{INDEX_NAME}' created successfully")
    except redis.ResponseError as e:
        logger.error(f"Failed to create index: {e}")
        raise

# Load tokenizer
logger.info("Loading tokenizer...")
with open('./corpus/tokeniser.pkl', 'rb') as f:
    tkns = pickle.load(f)
words_to_ids, ids_to_words = tkns['words_to_ids'], tkns['ids_to_words']
logger.info(f"Loaded tokenizer with {len(words_to_ids)} tokens")

# Load Word2Vec model
logger.info("Loading Word2Vec model...")
w2v = models.SkipGram(voc=len(words_to_ids), emb=EMBEDDING_DIM).to(dev)

# Find latest Word2Vec checkpoint
w2v_files = glob.glob('./checkpoints/*.w2v.pth')
if not w2v_files:
    raise FileNotFoundError("No Word2Vec checkpoints found")
w2v_checkpoint = max(w2v_files, key=os.path.getctime)
logger.info(f"Found latest Word2Vec checkpoint: {w2v_checkpoint}")

try:
    checkpoint = torch.load(w2v_checkpoint, map_location=dev)
    w2v.load_state_dict(checkpoint)
    w2v.eval()
    logger.info("Successfully loaded Word2Vec checkpoint")
except Exception as e:
    logger.error(f"Failed to load Word2Vec checkpoint: {e}")
    raise

# Load Two Towers model
logger.info("Loading Two Towers model...")
two = models.Towers(emb=EMBEDDING_DIM).to(dev)

# Find latest Two Towers checkpoint
two_files = glob.glob('./checkpoints/*.two.pth')
if not two_files:
    raise FileNotFoundError("No Two Towers checkpoints found")
two_checkpoint = max(two_files, key=os.path.getctime)
logger.info(f"Found latest Two Towers checkpoint: {two_checkpoint}")

try:
    checkpoint = torch.load(two_checkpoint, map_location=dev)
    if 'emb.weight' in checkpoint:
        raise ValueError("This appears to be a Word2Vec checkpoint, not a Two Towers checkpoint")
    two.load_state_dict(checkpoint)
    two.eval()
    logger.info("Successfully loaded Two Towers checkpoint")
except Exception as e:
    logger.error(f"Failed to load Two Towers checkpoint: {e}")
    raise

# Initialize dataset for tokenization
ds = dataset.Triplets(w2v.emb, words_to_ids)

def process_query(query_text):
    """Process a query and return similar documents."""
    try:
        # Get query embedding
        query_emb = ds.to_emb(query_text)
        if query_emb is None:
            logger.warning("Could not generate embedding for query")
            return None
        
        with torch.no_grad():
            query_emb = two.qry(query_emb)
            # Normalize the embedding
            query_emb = torch.nn.functional.normalize(query_emb, p=2, dim=0)
            query_emb = query_emb.detach().cpu().numpy().astype(np.float32)
        
        # Perform ANN search
        res = r.execute_command(
            "FT.SEARCH", INDEX_NAME,
            f"*=>[KNN {args.top_k} @embedding $vec AS dist]",
            "RETURN", 2, "text", "dist",
            "PARAMS", 2, "vec", query_emb.tobytes(),
            "DIALECT", 2
        )
        
        if len(res) <= 1:
            logger.info("No results found")
            return []
        
        results = []
        for i in range(1, len(res)-1, 2):
            doc_id = res[i].decode('utf-8')
            doc_fields = res[i+1]
            
            if not isinstance(doc_fields, list) or len(doc_fields) < 2:
                continue
                
            text = dist = None
            for j in range(0, len(doc_fields), 2):
                key = doc_fields[j]
                value = doc_fields[j+1]
                if key == b'text':
                    text = value.decode('utf-8', errors='ignore')
                elif key == b'dist':
                    try:
                        dist = float(value)
                        # Convert L2 distance to similarity score (0-100%)
                        # Using exponential decay: similarity = 100 * e^(-distance)
                        similarity = 100 * np.exp(-dist)
                    except (ValueError, TypeError) as e:
                        logger.error(f"Error converting distance for doc {doc_id}: {e}")
                        continue
            
            if dist is not None:
                results.append((similarity, text if text is not None else '[No text found]'))
                logger.debug(f"Doc {doc_id}: dist={dist:.4f}, similarity={similarity:.1f}%")
        
        # Sort by similarity (descending)
        return sorted(results, key=lambda x: x[0], reverse=True)
    
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return None

def main():
    logger.info("Ready to process queries. Type 'exit' to quit.")
    print("\nEnter your query (or 'exit' to quit). Try queries like:")
    print("- walgreens revenue")
    print("- starbucks store")
    print("- marriage act")
    print()
    
    while True:
        try:
            query = input("Enter your query: ").strip()
            if query.lower() == 'exit':
                break
            if not query:
                continue
                
            results = process_query(query)
            if not results:
                print("No results found.")
                continue
                
            print(f"\nTop {args.top_k} results:")
            for idx, (similarity, text) in enumerate(results, 1):
                print(f"\nRank {idx}: Similarity={similarity:.1f}%")
                print(f"Text: {text}")
                print("-" * 80)
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            continue

if __name__ == "__main__":
    main() 