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
def top_sim(w2v, sample):
  w2v.to(dev)
  sample = torch.tensor([words_to_ids[sample]]).to(dev)
  emb = w2v.emb(sample).squeeze()
  sim_cos = torch.nn.functional.cosine_similarity(emb, w2v.emb.weight, dim=1)
  top_v, top_i = torch.topk(sim_cos, k=4)
  return [(ids_to_words[i.item()], round(v.item(), 4)) for v, i in zip(top_v[1:], top_i[1:])]


#
#
#
if __name__ == '__main__':
  w2v = models.SkipGram(voc=len(words_to_ids), emb=128)
  w2v.load_state_dict(torch.load('./checkpoints/3.xjrg.w2v.pth'))
  print('dog',      top_sim(w2v, 'dog')[0])
  print('cat',      top_sim(w2v, 'cat')[0])
  print('law',      top_sim(w2v, 'law')[0])
  print('bee',      top_sim(w2v, 'bee')[0])
  print('tree',     top_sim(w2v, 'tree')[0])
  print('car',      top_sim(w2v, 'car')[0])
  print('house',    top_sim(w2v, 'house')[0])
  print('computer', top_sim(w2v, 'computer')[0])
  print('water',    top_sim(w2v, 'water')[0])
  print('food',     top_sim(w2v, 'food')[0])
  print('music',    top_sim(w2v, 'music')[0])
  print('senator',  top_sim(w2v, 'senator')[0])
