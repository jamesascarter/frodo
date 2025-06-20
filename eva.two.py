#
#
#
import torch
import models
import pickle


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

w2v = models.SkipGram(voc=len(words_to_ids), emb=128).to(dev)
w2v.load_state_dict(torch.load(w2v_checkpoint))


#
#
#

two_files = glob.glob('./checkpoints/*.two.pth')
if not two_files:
    raise FileNotFoundError("No Two checkpoints found")
two_checkpoint = max(two_files, key=os.path.getctime)
logger.info(f"Found latest Word2Vec checkpoint: {two_checkpoint}")

two = models.Towers(emb=128).to(dev)
two.load_state_dict(torch.load(two_checkpoint))


#
#
#
qry = torch.tensor([words_to_ids[w] for w in 'what animal bark'.split(' ')]).to(dev)
doc = torch.stack([torch.tensor([words_to_ids[w] for w in x.split(' ')]) for x in ['dog cute', 'cat meows', 'computers fast']]).to(dev)


#
#
#
qry = two.qry(w2v.emb(qry).mean(dim=0))
doc = two.doc(w2v.emb(doc).mean(dim=1))


#
#
#
res = torch.nn.functional.cosine_similarity(qry, doc)
print(res)
