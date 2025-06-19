import torch
import pickle
import models
import numpy as np

# Load the tokenizer
with open('./corpus/tokeniser.pkl', 'rb') as f:
    tkns = pickle.load(f)
words_to_ids, ids_to_words = tkns['words_to_ids'], tkns['ids_to_words']

# Load the model
w2v = models.SkipGram(voc=len(words_to_ids), emb=128)
w2v.load_state_dict(torch.load("./checkpoints/2025_06_19__11_42_58.4.70000.w2v.pth", map_location='cpu'))

print("Model loaded successfully")
print(f"Vocabulary size: {len(words_to_ids)}")

# Check embedding weights
emb_weights = w2v.emb.weight.data
print(f"Embedding weights shape: {emb_weights.shape}")

# Check for NaN values
nan_count = torch.isnan(emb_weights).sum().item()
print(f"NaN values in embeddings: {nan_count}")

# Check for infinite values
inf_count = torch.isinf(emb_weights).sum().item()
print(f"Infinite values in embeddings: {inf_count}")

# Check weight statistics
print(f"Embedding weights - Min: {emb_weights.min().item():.6f}, Max: {emb_weights.max().item():.6f}")
print(f"Embedding weights - Mean: {emb_weights.mean().item():.6f}, Std: {emb_weights.std().item():.6f}")

# Check if any entire rows are NaN
nan_rows = torch.isnan(emb_weights).any(dim=1)
nan_row_count = nan_rows.sum().item()
print(f"Rows with any NaN values: {nan_row_count}")

# Show some examples of problematic words
if nan_row_count > 0:
    print("\nExamples of words with NaN embeddings:")
    nan_indices = torch.where(nan_rows)[0]
    for i, idx in enumerate(nan_indices[:10]):  # Show first 10
        word = ids_to_words[idx.item()]
        print(f"  {word} (ID: {idx.item()})")

# Check linear layer weights
ffw_weights = w2v.ffw.weight.data
print(f"\nLinear layer weights shape: {ffw_weights.shape}")
print(f"Linear layer - Min: {ffw_weights.min().item():.6f}, Max: {ffw_weights.max().item():.6f}")
print(f"Linear layer - Mean: {ffw_weights.mean().item():.6f}, Std: {ffw_weights.std().item():.6f}")

ffw_nan_count = torch.isnan(ffw_weights).sum().item()
print(f"NaN values in linear layer: {ffw_nan_count}") 