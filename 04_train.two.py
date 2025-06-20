#
#
#
import torch
import models
import pickle
import dataset
import datetime
import wandb
import tqdm
import argparse
import os
import glob
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#
#
#
# Parse command line arguments
parser = argparse.ArgumentParser(description='Train Towers model')
parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train (default: 1)')
parser.add_argument('--batch_size', type=int, default=256, help='Batch size (default: 256)')
parser.add_argument('--learning_rate', type=float, default=0.0007165, help='Learning rate (default: 0.003)')
parser.add_argument('--margin', type=float, default=0.129, help='Margin for triplet loss (default: 0.3)')
parser.add_argument('--embedding_dim', type=int, default=128, help='Embedding dimension (default: 128)')
args = parser.parse_args()

torch.manual_seed(42)
ts = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')


#
#
#
with open('./corpus/tokeniser.pkl', 'rb') as f: tkns = pickle.load(f)
words_to_ids, ids_to_words = tkns['words_to_ids'], tkns['ids_to_words']
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#
#
#
w2v_files = glob.glob('./checkpoints/*.w2v.pth')
if not w2v_files:
    raise FileNotFoundError("No Word2Vec checkpoints found")
w2v_checkpoint = max(w2v_files, key=os.path.getctime)
logger.info(f"Found latest Word2Vec checkpoint: {w2v_checkpoint}")

# Load w2v model (assuming it's defined in models)
w2v = models.SkipGram(voc=len(words_to_ids), emb=args.embedding_dim)  # You'll need to adjust this based on your actual model
w2v.load_state_dict(torch.load(w2v_checkpoint))


#
#
#
ds = dataset.Triplets(w2v.emb, words_to_ids)
dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True)


#
#
#
two = models.Towers(emb=args.embedding_dim).to(dev)
torch.save(two.state_dict(), f'./checkpoints/{ts}.0.0.two.pth')
print('two:', sum(p.numel() for p in two.parameters())) # 66,048
opt = torch.optim.Adam(two.parameters(), lr=args.learning_rate)

# Initialize wandb with sweep config
wandb.init(project='mlx6-week-02-two')


#
#
#
for epoch in range(args.epochs):
  prgs = tqdm.tqdm(dl, desc=f"Epoch {epoch + 1}", leave=False)
  for idx, (qry, pos, neg) in enumerate(prgs):
    qry, pos, neg = qry.to(dev), pos.to(dev), neg.to(dev)
    loss = two(qry, pos, neg, mrg=args.margin)
    opt.zero_grad()
    loss.backward()
    opt.step()
    wandb.log({'loss': loss.item()})
    if idx % 50 == 0: torch.save(two.state_dict(), f'./checkpoints/{ts}.{epoch}.{idx}.two.pth')


#
#
#
wandb.finish()