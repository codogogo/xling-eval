import argparse
import util
import os

parser = argparse.ArgumentParser(description="Running embeddings CONVERTER, from serialized to textual format.")
parser.add_argument("vocab_path", type=str, help="Path to the file in which to serialize the embedding space vocabulary")
parser.add_argument("vectors_path", type=str, help="Path to the file in which to serialize the embedding vectors")
parser.add_argument("text_embs_path", type=str, help="Path where the word embeddings in the textual format will be serialized")

args = parser.parse_args()

if not os.path.isfile(args.vocab_path):
	print("Error: File with the serialized vocabulary not found!")
	exit(code = 1)

if not os.path.isfile(args.vectors_path):
	print("Error: File with the serialized vectord not found!")
	exit(code = 1)

if not os.path.isdir(os.path.dirname(args.text_embs_path)):
	print("Error: Output directory for the textual embeddings output not found!")
	exit(code = 1)

print("Deserializing...")
embs, _, voc, _ = util.deserialize_embs(args.vocab_path, args.vectors_path, False, False)

print("Storing in textual format...")
util.write_embs(args.text_embs_path, voc, embs)
print("I'm all done here, ciao bella!")