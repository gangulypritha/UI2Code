import argparse
import os

parser = argparse.ArgumentParser(description='Generate the vocabulary file based on the specifed dataset')

parser.add_argument("--data_path", type=str,help="Datapath")

args = parser.parse_args()

data_path = args.data_path
vocab_output_path = os.path.join(os.path.dirname(data_path), "vocab.txt")

all_tokens = dict() # dict used as ordered set since it preserves order
for file in os.listdir(data_path):
    stem = file.split(".")[0]
    suffix = "." + file.split(".")[1]

    if suffix == ".gui":
        with open(os.path.join(data_path,file), "r") as reader:
            raw_data = reader.read()
            data = raw_data.replace('\n', ' ').replace(', ', ' , ').split(' ')
            data.remove('')
            for el in data:
                all_tokens[el] = el


# write the set of all tokens to a vocab file
print(f'Writing vocab with {len(all_tokens)} tokens')

with open(vocab_output_path, "w") as writer:
    writer.write(" ".join(all_tokens))
