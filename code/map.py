import argparse
import util
import projection
import numpy as np
import os
from datetime import datetime
import time
import pickle

parser = argparse.ArgumentParser(description="Running YACLE, a tool for building cross-lingual embedding spaces.")
parser.add_argument("embs_src", type=str, help="Path to the file containing the serialized pre-trained source language embeddings")
parser.add_argument("vocab_src", type=str, help="Path to the file containing the serialized source language vocabulary")
parser.add_argument("embs_trg", type=str, help="Path to the file containing the serialized pre-trained target language embeddings")
parser.add_argument("vocab_trg", type=str, help="Path to the file containing the serialized target language vocabulary")
parser.add_argument("output", type=str, help="Path to the directory where the bilingual embeddings are to be stored")
parser.add_argument("-m", "--model", type=str, default="p", help="Mapping model to run: simple Procrustes mapping (PROC, value 'p'), Procrustes with Bootstrapping (PROC-B, value 'b'), or Cannonical Correlation Analysis mapping (CCA, value 'c')")
parser.add_argument("-d", "--trans_dict", type=str, help="Translation dictionary with word translations to be used for computing the projection matrix. If not provided identically spelled words between vocabularies are used.")
parser.add_argument("--lang_src", type=str, help="The source language name (used only for the filenames of output files)") 
parser.add_argument("--lang_trg", type=str, help="The target language name (used only for the filenames of output files)")
args = parser.parse_args()

if not os.path.isfile(args.embs_src):
	print("Error: File with the source language embeddings not found!")
	exit(code = 1)

if not os.path.isfile(args.vocab_src):
	print("Error: File with the source language vocabulary not found!")
	exit(code = 1)

if not os.path.isfile(args.embs_trg):
	print("Error: File with the target language embeddings not found!")
	exit(code = 1)

if not os.path.isfile(args.vocab_trg):
	print("Error: File with the target language vocabulary not found!")
	exit(code = 1)

if not os.path.isdir(os.path.dirname(args.output)):
	print("Error: Output directory not found.")
	exit(code = 1)

if args.trans_dict and not os.path.isfile(args.trans_dict):
    print("Error: Translation dictionary file not found.")
    exit(code = 1)

print("Loading source embeddings and vocabulary...")
src_embs = np.load(args.embs_src)
vocab_src = pickle.load(open(args.vocab_src,"rb"))

print("Loading target embeddings and vocabulary...")
trg_embs = np.load(args.embs_trg)
vocab_trg = pickle.load(open(args.vocab_trg,"rb"))

model = args.model
if model not in ["p", "b", "c", "r"]:
  print("Error: Unknown mapping/projection model.")
  exit(code = 1)

print("Loading translation dictionary...")
trans_dict = [x.lower().split("\t") for x in util.load_lines(args.trans_dict)]

lang_src = args.lang_src if args.lang_src else "src"
lang_trg = args.lang_trg if args.lang_trg else "trg"

output = args.output if args.output.strip()[-1] == '/' else args.output + "/"

print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + " Inducing the mapping and creating a bilingual embedding space")

if model == "p":
  embs_src_shared, proj_mat, _ = projection.project_proc(vocab_src, src_embs, vocab_trg, trg_embs, trans_dict)
  embs_trg_shared = trg_embs

elif model == "b":
  embs_src_shared, embs_trg_shared = projection.project_proc_bootstrap(vocab_src, src_embs, vocab_trg, trg_embs, trans_dict)

elif model == "c":
  embs_src_shared, embs_trg_shared, _ = projection.project_cca(vocab_src, src_embs, vocab_trg, trg_embs, trans_dict)

elif model == "r":
  embs_src_shared, embs_trg_shared, _ = projection.project_proc_bootstrap_reproduce(vocab_src, src_embs, vocab_trg, trg_embs, trans_dict)
  

print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + " Serializing projected source language embeddings...")
util.serialize_embs(output + lang_src + "-" + lang_trg + "." + lang_src + ".vocab", output + lang_src + "-" + lang_trg + "." + lang_src + ".vectors", vocab_src, embs_src_shared, emb_norm = False, vocab_inv = False)

print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + " Serializing target language embeddings...")
util.serialize_embs(output + lang_src + "-" + lang_trg + "." + lang_trg + ".vocab", output + lang_src + "-" + lang_trg + "." + lang_trg + ".vectors", vocab_trg, embs_trg_shared, emb_norm = False, vocab_inv = False)

if model == "p":
  print("Saving projection matrix...")
  np.save(output + lang_src + "-" + lang_trg + ".proj", proj_mat)

print(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S') + " All done. I'm out of here, ciao bella!")

   


