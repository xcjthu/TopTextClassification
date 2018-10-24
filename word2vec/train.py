import argparse
import os
import fasttext

parser = argparse.ArgumentParser()
parser.add_argument('--output', '-o')
parser.add_argument('--input', '-i')
parser.add_argument("--dim", "-d")
args = parser.parse_args()

input_file_path = args.input
output_file_path = args.output
dim = int(args.dim)

os.makedirs(output_file_path, exist_ok=True)

model = fasttext.skipgram(input_file_path, os.path.join(output_file_path, "model"), dim=dim, thread=12)
