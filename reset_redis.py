import redis
import os
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

# Drop existing index if it exists
try:
    r.execute_command(f"FT.DROPINDEX {INDEX_NAME}")
    logger.info(f"Dropped existing index '{INDEX_NAME}'")
except redis.ResponseError as e:
    if "Unknown Index name" not in str(e):
        logger.error(f"Error dropping index: {e}")
        raise
    logger.info(f"No existing index '{INDEX_NAME}' to drop")

# Clear entire database (much faster than deleting keys one by one)
logger.info("Clearing entire Redis database...")
r.flushdb()
logger.info("Database cleared successfully")

# Create new index with proper settings
try:
    r.execute_command(
        "FT.CREATE", INDEX_NAME, "ON", "HASH", 
        "PREFIX", "1", "doc:",
        "SCHEMA",
        "text", "TEXT",
        "doc_id", "TAG",
        "embedding", "VECTOR", "FLAT", "6",
        "TYPE", "FLOAT32", "DIM", str(EMBEDDING_DIM), "DISTANCE_METRIC", "L2"
    )
    logger.info(f"Created new index '{INDEX_NAME}' with L2 distance metric")
except redis.ResponseError as e:
    logger.error(f"Failed to create index: {e}")
    raise

# Delete all document keys
pattern = "doc:*"
cursor = 0
deleted = 0

logger.info("Deleting existing documents...")
while True:
    cursor, keys = r.scan(cursor, match=pattern, count=100)
    if keys:
        r.delete(*keys)
        deleted += len(keys)
    if cursor == 0:
        break

logger.info(f"Deleted {deleted} documents")
logger.info("Redis reset complete. Ready for new documents.") 