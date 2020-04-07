import numpy as np
import cca
import util

def get_seeds(vocab_dict_src, vocab_dict_trg, n = 5000):
  allmatches = [k for k in vocab_dict_src if k in vocab_dict_trg]
  allmatches.sort(key = lambda x: vocab_dict_src[x] + vocab_dict_trg[x])
  return allmatches[:n]

def build_matrices(vocab_dict_src, vocab_dict_trg, embs_src, embs_trg, trans_dict = None, num_same = 5000):
  src_mat = []
  trg_mat = []
  if trans_dict:
    for sw, tw in trans_dict:
      if sw in vocab_dict_src and tw in vocab_dict_trg:
        src_mat.append(embs_src[vocab_dict_src[sw]])
        trg_mat.append(embs_trg[vocab_dict_trg[tw]])
  else: 
    seeds = get_seeds(vocab_dict_src, vocab_dict_trg, n = num_same)
    for s in seeds:
      src_mat.append(embs_src[vocab_dict_src[s]])
      trg_mat.append(embs_trg[vocab_dict_trg[s]])
  return np.array(src_mat, dtype=np.float32), np.array(trg_mat, dtype=np.float32)

def project_pinv(vocab_dict_src, embs_src, vocab_dict_trg, embs_trg, trans_dict = None):
  src_mat, trg_mat = build_matrices(vocab_dict_src, vocab_dict_trg, embs_src, embs_trg, trans_dict)
  proj_mat = np.dot(np.linalg.pinv(src_mat), trg_mat)
  return np.dot(embs_src, proj_mat), proj_mat

def project_cca(vocab_dict_src, embs_src, vocab_dict_trg, embs_trg, trans_dict = None):
  src_mat, trg_mat = build_matrices(vocab_dict_src, vocab_dict_trg, embs_src, embs_trg, trans_dict)
  corr_an = cca.CCA(src_mat, trg_mat, min(src_mat.shape[1], trg_mat.shape[1]))
  corr_an.correlate(sklearn = False)
  proj_src, proj_trg = corr_an.transform(embs_src, embs_trg)
  return proj_src, proj_trg, corr_an

def project_proc(vocab_dict_src, embs_src, vocab_dict_trg, embs_trg, trans_dict = None):
  src_mat, trg_mat = build_matrices(vocab_dict_src, vocab_dict_trg, embs_src, embs_trg, trans_dict)
  product = np.matmul(src_mat.transpose(), trg_mat)
  U, s, V = np.linalg.svd(product)
  proj_mat = np.matmul(U, V)

  embs_src_projected = np.matmul(embs_src, proj_mat)  
  return embs_src_projected, proj_mat, src_mat.shape[0] 
  
def project_proc_bootstrap(vocab_dict_src, embs_src, vocab_dict_trg, embs_trg, trans_dict = None, growth_rate = 1.5, limit = 10000):
  vocab_dict_src_inv = {v : k for k, v in vocab_dict_src.items()}
  vocab_dict_trg_inv = {v : k for k, v in vocab_dict_trg.items()} 
  cnt = 0

  orig_src_norm = util.mat_normalize(embs_src, norm_order=2, axis=1)
  orig_trg_norm = util.mat_normalize(embs_trg, norm_order=2, axis=1) 

  size = 0
  while True:
    cnt += 1
    print("Boostrap iteration: " + str(cnt))
    
    embs_src_projected, _, size1 = project_proc(vocab_dict_src, embs_src, vocab_dict_trg, embs_trg, trans_dict)
    embs_trg_projected, _, size2 = project_proc(vocab_dict_trg, embs_trg, vocab_dict_src, embs_src, [(x[1], x[0]) for x in trans_dict])
    
    if size1 < 1.01 * size or size1 >= limit:
      break
    else:
      size = size1

    proj_src_norm = util.mat_normalize(embs_src_projected, norm_order=2, axis=1)
    proj_trg_norm = util.mat_normalize(embs_trg_projected, norm_order=2, axis=1)
    
    sims_ind_src_trg = util.big_matrix_multiplication(proj_src_norm, orig_trg_norm.transpose(), lambda x: np.argmax(x, axis = 1), chunk_size = 30000)
    sims_ind_trg_src = util.big_matrix_multiplication(proj_trg_norm, orig_src_norm.transpose(), lambda x: np.argmax(x, axis = 1), chunk_size = 30000)
  
    matches = [i for i in range(len(sims_ind_src_trg)) if sims_ind_trg_src[sims_ind_src_trg[i]] == i]

    rank_pairs = [(m, sims_ind_src_trg[m]) for m in matches]
    rank_pairs.sort(key=lambda x: x[0] + x[1])
    cnt = min(int(growth_rate * len(trans_dict)), limit)
    
    if cnt < len(rank_pairs):
      rank_pairs = rank_pairs[:cnt]

    new_trans_dict = [(vocab_dict_src_inv[m[0]], vocab_dict_trg_inv[m[1]]) for m in rank_pairs]
    print(new_trans_dict)
    print("Dict size for next iteration: " + str(len(new_trans_dict)))
    trans_dict = new_trans_dict

  return embs_src_projected, embs_trg

def project_proc_bootstrap_reproduce(vocab_dict_src, embs_src, vocab_dict_trg, embs_trg, trans_dict = None, growth_rate = 1.5, limit = 10000):
  vocab_dict_src_inv = {v : k for k, v in vocab_dict_src.items()}
  vocab_dict_trg_inv = {v : k for k, v in vocab_dict_trg.items()} 
  cntr = 0

  orig_src_norm = util.mat_normalize(embs_src, norm_order=2, axis=1)
  orig_trg_norm = util.mat_normalize(embs_trg, norm_order=2, axis=1) 

  size = 0
  orig_trans_dict = trans_dict
  while True:
    cntr += 1
    print("Boostrap iteration: " + str(cntr))
    
    embs_src_projected, _, size1 = project_proc(vocab_dict_src, embs_src, vocab_dict_trg, embs_trg, trans_dict)
    embs_trg_projected, _, size2 = project_proc(vocab_dict_trg, embs_trg, vocab_dict_src, embs_src, [(x[1], x[0]) for x in trans_dict])
    
    if size1 < 1.01 * size:
      break
    else:
      size = size1

    proj_src_norm = util.mat_normalize(embs_src_projected, norm_order=2, axis=1)
    proj_trg_norm = util.mat_normalize(embs_trg_projected, norm_order=2, axis=1)

    if cntr == 2:
      break

    sims_ind_src_trg = util.big_matrix_multiplication(proj_src_norm, orig_trg_norm.transpose(), lambda x: np.argmax(x, axis = 1), chunk_size = 30000)
    sims_ind_trg_src = util.big_matrix_multiplication(proj_trg_norm, orig_src_norm.transpose(), lambda x: np.argmax(x, axis = 1), chunk_size = 30000)
  
    matches = [i for i in range(len(sims_ind_src_trg)) if sims_ind_trg_src[sims_ind_src_trg[i]] == i]

    rank_pairs = [(m, sims_ind_src_trg[m]) for m in matches]
    rank_pairs.sort(key=lambda x: x[0] + x[1])
    new_trans_dict = [(vocab_dict_src_inv[m[0]], vocab_dict_trg_inv[m[1]]) for m in rank_pairs]	

    cnt = limit - len(orig_trans_dict)

    fin_tr_dict = []
    fin_tr_dict.extend(orig_trans_dict)
    fin_tr_dict.extend(new_trans_dict[:cnt])     

    print("Dict size for next iteration: " + str(len(fin_tr_dict)))
    trans_dict = fin_tr_dict

  return embs_src_projected, embs_trg
