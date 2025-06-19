#
#
#
import collections
import pickle


#
#
#
with open('./corpus/text8.txt') as f: text8: str = f.read()
with open('./corpus/msmarco.txt') as f: msmarco: str = f.read()


#
#
#
def preprocess(text: str) -> list[str]:
  text = text.lower()
  text = text.replace('.',  ' <PERIOD> ')
  text = text.replace(',',  ' <COMMA> ')
  text = text.replace('"',  ' <QUOTATION_MARK> ')
  text = text.replace('“',  ' <QUOTATION_MARK> ')
  text = text.replace('”',  ' <QUOTATION_MARK> ')
  text = text.replace(';',  ' <SEMICOLON> ')
  text = text.replace('!',  ' <EXCLAMATION_MARK> ')
  text = text.replace('?',  ' <QUESTION_MARK> ')
  text = text.replace('(',  ' <LEFT_PAREN> ')
  text = text.replace(')',  ' <RIGHT_PAREN> ')
  text = text.replace('--', ' <HYPHENS> ')
  text = text.replace('?',  ' <QUESTION_MARK> ')
  text = text.replace(':',  ' <COLON> ')
  text = text.replace("'",  ' <APOSTROPHE> ')
  text = text.replace("’",  ' <APOSTROPHE> ')
  words = text.split()
  stats = collections.Counter(words)
  words = [word for word in words if stats[word] > 5]
  return words

def filter_corpus_to_top_words(corpus: list[str], top_n: int = 50000) -> list[str]:
  """Filter corpus to only include the top N most frequent words."""
  # Count word frequencies in the corpus
  word_counts = collections.Counter(corpus)
  
  # Get the top N most frequent words
  top_words = set(word for word, count in word_counts.most_common(top_n))
  
  # Filter corpus to only include top words
  filtered_corpus = [word for word in corpus if word in top_words]
  
  return filtered_corpus

#
#
#
def create_lookup_tables(words: list[str]) -> tuple[dict[str, int], dict[int, str]]:
  word_counts = collections.Counter(words)
  vocab = sorted(word_counts, key=lambda k: word_counts.get(k), reverse=True)
  int_to_vocab = {ii+1: word for ii, word in enumerate(vocab)}
  int_to_vocab[0] = '<PAD>'
  vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}
  return vocab_to_int, int_to_vocab


#
#
#
corpus: list[str] = preprocess(text8 + msmarco)
print("INITIAL CORPUS SIZE:", len(corpus))
print("INITIAL VOCAB SIZE:", len(set(corpus)))

corpus = filter_corpus_to_top_words(corpus, 50000)
print("FILTERED CORPUS SIZE:", len(corpus))
print("FILTERED VOCAB SIZE:", len(set(corpus)))


words_to_ids, ids_to_words = create_lookup_tables(corpus)
tokeniser = tokeniser = { "words_to_ids": words_to_ids, "ids_to_words": ids_to_words }
tokens: list[int] = [words_to_ids[word] for word in corpus]
with open('./corpus/tokens.txt', 'w', encoding='utf-8') as f: f.write('\n'.join(map(str, tokens)))
with open('./corpus/tokeniser.pkl', 'wb') as f: pickle.dump(tokeniser, f)


#
#
#
print("FINAL VOCAB:", len(words_to_ids))
