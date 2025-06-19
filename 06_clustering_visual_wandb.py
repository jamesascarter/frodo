import wandb
import numpy as np
from sklearn.manifold import TSNE
import torch
import pickle
import models

# 1. Load the tokenizer to get word mappings
with open('./corpus/tokeniser.pkl', 'rb') as f:
    tkns = pickle.load(f)
words_to_ids, ids_to_words = tkns['words_to_ids'], tkns['ids_to_words']

# 2. Load your trained PyTorch SkipGram model
w2v = models.SkipGram(voc=len(words_to_ids), emb=128)
w2v.load_state_dict(torch.load("./checkpoints/2025_06_19__11_42_58.4.70000.w2v.pth", map_location='cpu'))
w2v.eval()

# 3. Select a list of words to visualize (for example, top 500 most frequent words)
words = list(words_to_ids.keys())[:500]

# 4. Extract their vectors from the embedding layer
vectors = []
valid_words = []
for word in words:
    if word in words_to_ids:
        word_id = words_to_ids[word]
        with torch.no_grad():
            word_vector = w2v.emb(torch.tensor([word_id])).squeeze().numpy()
        vectors.append(word_vector)
        valid_words.append(word)

vectors = np.array(vectors)

# 5. Dimensionality reduction (t-SNE)
tsne = TSNE(n_components=2, random_state=42)
vectors_2d = tsne.fit_transform(vectors)

# 6. Initialize wandb
wandb.init(project="word2vec-embedding-clusters")

# 7. Create a wandb Table to log embeddings with labels
table = wandb.Table(columns=["word", "x", "y"])

for word, (x, y) in zip(valid_words, vectors_2d):
    table.add_data(word, x, y)

# 8. Log the Table as an embedding visualization
wandb.log({"word2vec_embedding_clusters": table})

# (Optional) You can also create a scatter plot manually if you want colors/clusters:
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], alpha=0.7)

for i, word in enumerate(valid_words):
    plt.annotate(word, (vectors_2d[i, 0], vectors_2d[i, 1]))

wandb.log({"embedding_tsne_plot": plt})
