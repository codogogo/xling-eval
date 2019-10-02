import util
import argparse
import os

parser = argparse.ArgumentParser(description="Running YACLE CONVERTER, a tool for more compact serialization of pretrained embeddings.")
parser.add_argument("text_embs_path", type=str, help="Path to the file word embeddings in the textual format")
parser.add_argument("vocab_path", type=str, help="Path to the file in which to serialize the embedding space vocabulary")
parser.add_argument("vectors_path", type=str, help="Path to the file in which to serialize the embedding vectors")
parser.add_argument("-n", "--topn", type=int, help="Number of top vocabulary words to serialize. (default is the whole vocabulary)")
parser.add_argument("-d", "--dim", type=int, help="The size of the text embeddings. (default is 300)")
args = parser.parse_args()

if not os.path.isfile(args.text_embs_path):
	print("Error: File with the textual embeddings not found!")
	exit(code = 1)


if not os.path.isdir(os.path.dirname(args.vocab_path)):
	print("Error: Output directory for the vocabulary file serialization not found!")
	exit(code = 1)

if not os.path.isdir(os.path.dirname(args.vectors_path)):
	print("Error: Output directory for the vectors file serialization not found!")
	exit(code = 1)

d = 300 if not args.dim else args.dim
util.load_and_serialize_embs(args.text_embs_path, args.vocab_path, args.vectors_path, args.topn, dimension = d) 