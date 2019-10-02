import numpy as np

def most_similar(word, vocab_src, src_embs_norm, trg_embs_norm, vocab_trg_inv, num = 100):
  if word not in vocab_src:
    print("Word not found in vocabulary: " + word)
    return None
  word_emb = src_embs_norm[vocab_src[word]]
  sims = np.dot(word_emb, np.transpose(trg_embs_norm))
  inds = np.argsort(sims)[-1 * num :][::-1]
  scores = [sims[ind] for ind in inds]
  words = [vocab_trg_inv[ind] for ind in inds]
  return list(zip(words, scores))

def most_similar_index(word_src, word_trg, vocab_src, vocab_trg, src_embs_norm, trg_embs_norm):
  if word_src not in vocab_src and word_src.lower() not in vocab_src:
    print("Word not found in vocabulary: " + word_src)
    return None
  if word_trg not in vocab_trg and word_trg.lower() not in vocab_trg:
    print("Word not found in vocabulary: " + word_trg)
    return None
  word_src_emb = src_embs_norm[vocab_src[word_src] if word_src in vocab_src else vocab_src[word_src.lower()]]
  sims = np.dot(word_src_emb, np.transpose(trg_embs_norm))
  inds = np.argsort(sims)[::-1]
  trg_ind = vocab_trg[word_trg] if word_trg in vocab_trg else vocab_trg[word_trg.lower()]
  return np.where(inds == trg_ind)[0][0] + 1

def similarity(word_src, word_trg, vocab_src, src_embs_norm, vocab_trg, trg_embs_norm):
  if word_src not in vocab_src: 
    print("Source language word not found in vocabulary: " + word_src)
    return None
  if word_trg not in vocab_trg: 
    print("Target language word not found in vocabulary: " + word_trg)
    return None
  return np.dot(src_embs_norm[vocab_src[word_src]], trg_embs_norm[vocab_trg[word_trg]])
  