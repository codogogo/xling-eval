import numpy as np
import codecs
import pickle
import util
from sys import stdin

def load_lines(path):
	return [l.strip() for l in list(codecs.open(path, "r", encoding = 'utf8', errors = 'replace').readlines())]

def write_lines(path, list, append = False):
	f = codecs.open(path,"a" if append else "w",encoding='utf8')
	for l in list:
		f.write(str(l) + "\n")
	f.close()

def write_text(path, text, append = False):
    f = codecs.open(path,"a" if append else "w",encoding='utf8')
    f.write(text + "\n")
    f.close()

def load_embs(path, topk = None, dimension = None):
  print(topk)
  print("Loading embeddings")
  vocab_dict = {}
  embeddings = []
  with codecs.open(path, encoding = 'utf8', errors = 'replace') as f:
      line = f.readline().strip().split()
      cntr = 1
      if len(line) == 2:
        vocab_size = int(line[0])
        if not dimension: 
          dimension = int(line[1])
      else: 
        if not dimension or (dimension and len(line[1:]) == dimension):
          vocab_dict[line[0].strip()] = len(vocab_dict)
          embeddings.append(np.array(line[1:], dtype=np.float32))
        if not dimension:
          dimension = len(line) - 1
      print("Vector dimensions: " + str(dimension))
      while line: 
        line = f.readline().strip().split() 
        if (not line):
          print("Loaded " + str(cntr) + " vectors.") 
          break

        if line[0].strip() == "":
          continue
        
        cntr += 1
        if cntr % 1000 == 0:
          print(cntr)

        if len(line[1:]) == dimension:
          if (line[0].strip().lower() not in vocab_dict): 
              vocab_dict[line[0].strip().lower()] = len(vocab_dict) 
              embeddings.append(np.array(line[1:], dtype=np.float32))
        else: 
          print("Error in the embeddings file, line " + str(cntr) + 
                             ": unexpected vector length (expected " + str(dimension) + 
                             " got " + str(len(np.array(line[1:]))) + " for word '" + line[0] + "'")
        
        if (topk and cntr >= topk): 
          print("Loaded " + str(cntr) + " vectors.") 
          break           

  embeddings = np.array(embeddings, dtype=np.float32)
  print(len(vocab_dict), str(embeddings.shape))
  return vocab_dict, embeddings

def write_embs(path, vocab, embs):
  if (len(vocab)) != embs.shape[0]:
    raise ValueError("Something wrong: vocabulary and emb. matrix don't have the same number of entries")

  f = codecs.open(path,"w",encoding='utf8')
  inv_dict = {v : k for k, v in vocab.items()}
  for i in range(len(vocab)):
    if i % 1000 == 0: 
      print("Writing line " + str(i))
    f.write(inv_dict[i] + " " + " ".join([str(round(float(x), 6)) for x in embs[i]]) + " \n")
  f.close()

def serialize_embs(path_vocab, path_embs, vocab_dict, embeddings, emb_norm = True, vocab_inv = True): 
  print("Serializing vocabulary...")
  np.save(open(path_embs,"wb+"),embeddings)
  print("Serializing embs...") 
  pickle.dump(vocab_dict, open(path_vocab,"wb+"))
  if emb_norm: 
    norm_embs = util.mat_normalize(embeddings)
    np.save(open(path_embs + ".norm","wb+"),norm_embs)
  if vocab_inv:
    pickle.dump({v : k for k, v in vocab_dict.items()}, open(path_vocab + ".inv","wb+"))

def deser_simple(path_vocab, path_embs): 
  embeddings = np.load(path_embs)
  vocab_dict = pickle.load(open(path_vocab,"rb"))
  return vocab_dict, embeddings

def deserialize_embs(path_vocab, path_embs, emb_norm = True, vocab_inv = True): 
  embeddings = np.load(path_embs)
  vocab_dict = pickle.load(open(path_vocab,"rb"))
  if emb_norm: 
    norm_embs = np.load(path_embs + ".norm")
  else:
    norm_embs = util.mat_normalize(embeddings)
  if vocab_inv:
    vocab_dict_inv = pickle.load(open(path_vocab + ".inv","rb"))
  else:
    vocab_dict_inv = {v : k for k, v in vocab_dict.items()}
  
  return embeddings, norm_embs, vocab_dict, vocab_dict_inv
  #return embeddings, vocab_dict

def load_and_serialize_embs(path_load, path_vocab_ser, path_embs_ser, topk = None, dimension = None):
  voc, embs = load_embs(path_load, topk, dimension)
  print(len(voc))
  print(embs.shape)
  serialize_embs(path_vocab_ser, path_embs_ser, voc, embs, emb_norm = False, vocab_inv = False)

def check_in_vocabulary(check_dict, vocabs, lower = True):
  #print(check_dict)
  for lang, word in check_dict.items():
    #print(lang, word)
    for (ef, voc) in vocabs[lang]:
      #print(ef)
      #stdin.readline()
      if not word in voc and (not lower or (lower and not word.lower() in voc)):
        #print("Not found in vocabulary " + (lang) + ": "  + word + " in " + ef)
        return False
  return True

def prefix_lang(vocab_dict, lang, delimiter = '_'):
  return { lang.value + delimiter + k : v for k, v in vocab_dict.items()}

def mat_normalize(mat, norm_order=2, axis=1):
  return mat / np.transpose([np.linalg.norm(mat, norm_order, axis)])

def big_matrix_multiplication(a, b, function_on_result, chunk_size = 100):
  result = []
  num_iters = a.shape[0] // chunk_size + (0 if a.shape[0] % chunk_size == 0 else 1)
  for i in range(num_iters):
    print("Batch multiplication iter: " + str(i+1))
    mul_batch = np.dot(a[i * chunk_size : (i+1) * chunk_size, :], b)
    res_batch = function_on_result(mul_batch)
    result.extend(res_batch)
  return result

def big_matrix_csls(a, b, csls_a, csls_b, chunk_size = 100):
  result = []
  num_iters = a.shape[0] // chunk_size + (0 if a.shape[0] % chunk_size == 0 else 1)
  for i in range(num_iters):
    size_a = chunk_size if i != (num_iters -1) else a.shape[0] % chunk_size
    size_b = b.shape[1]

    print("CSLS computation iter: " + str(i+1))
    csls_mat = np.zeros((size_a, b.shape[1])) + np.reshape(csls_a[i * chunk_size : i * chunk_size + size_a], (size_a, 1)) + np.reshape(csls_b, (1, size_b))
    mul_batch = np.dot(a[i * chunk_size : i * chunk_size + size_a, :], b)
    csls_batch = 2 * mul_batch - csls_mat
    res_batch = np.argmax(csls_batch, axis = 1)
    result.extend(res_batch)
  return result

