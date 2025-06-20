#
#
#
import torch
import pickle
import models


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
qry = w2v.emb(torch.tensor(words_to_ids['computer']).to(dev)).squeeze()
res = torch.nn.functional.cosine_similarity(qry, w2v.emb.weight)
top_v, top_i = torch.topk(res, k=4)
print([ids_to_words[i.item()] for i in top_i])
