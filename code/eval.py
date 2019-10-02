import util
import sims

import argparse
import util
import os
from datetime import datetime
import time
from sys import stdin
import sims

parser = argparse.ArgumentParser(description="Running the BLI evaluation script for cross-lingual word embedding spaces.")
parser.add_argument("test_set_path", type=str, help="Path to the file containing the test word pairs')")
parser.add_argument("yacle_embs_src", type=str, help="Path to the file containing the YACLE serialized, previously *PROJECTED*, source language embeddings (typically something like 'vectors_src.np')")
parser.add_argument("yacle_embs_trg", type=str, help="Path to the file containing the YACLE serialized target language embeddings (typically something like 'vectors_trg.np')")
parser.add_argument("vocab_src", type=str, help="Path to the file containing the YACLE serialized source language vocabulary dictionary (typically something like 'vocab_src.pkl')")
parser.add_argument("vocab_trg", type=str, help="Path to the file containing the YACLE serialized target language vocabulary dictionary (typically something like 'vocab_trg.pkl')")
args = parser.parse_args()

if not os.path.isfile(args.test_set_path):
	print("Error: File with the test set not found!")
	exit(code = 1)

if not os.path.isfile(args.yacle_embs_src):
	print("Error: File with the serialized projected source language embeddings not found!")
	exit(code = 1)

if not os.path.isfile(args.yacle_embs_trg):
	print("Error: File with the serialized target language embeddings not found!")
	exit(code = 1)

if not os.path.isfile(args.vocab_src):
	print("Error: File with the vocabulary dictionary of the source language not found!")
	exit(code = 1)

if not os.path.isfile(args.vocab_trg):
	print("Error: File with the vocabulary dictionary of the target language not found!")
	exit(code = 1)

print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + " Deserializing projected source language embeddings...")
embs_src, norm_embs_src, vocab_dict_src, vocab_dict_inv_src = util.deserialize_embs(args.vocab_src, args.yacle_embs_src, emb_norm = False, vocab_inv = False)

print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + " Deserializing target language embeddings...")
embs_trg, norm_embs_trg, vocab_dict_trg, vocab_dict_inv_trg = util.deserialize_embs(args.vocab_trg, args.yacle_embs_trg, emb_norm = False, vocab_inv = False)

positions = []
eval_dict_pairs = [x.lower().split("\t") for x in util.load_lines(args.test_set_path)]

cntr = 0
for ep in eval_dict_pairs:
  cntr += 1
  if cntr % 10 == 0:
    print(cntr)
  ind = sims.most_similar_index(ep[0].strip(), ep[1].strip(), vocab_dict_src, vocab_dict_trg, norm_embs_src, norm_embs_trg) 
  if ind:
    positions.append(ind)

p1 = len([p for p in positions if p == 1]) / len(positions)
p5 = len([p for p in positions if p <= 5]) / len(positions)
p10 = len([p for p in positions if p <= 10]) / len(positions)
mrr = sum([1.0/p for p in positions]) / len(positions)

print("Pairs evaluated: " + str(len(positions)))
print(positions)
print("P1: " + str(p1))
print("P5: " + str(p5))
print("P10: " + str(p10))
print("MRR: " + str(mrr))