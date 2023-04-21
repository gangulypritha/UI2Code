import argparse
import os
from math import floor

parser = argparse.ArgumentParser(description='Split the dataset into train, validation and test datasets')

parser.add_argument("--data_path", type=str, help="Datapath")

args = parser.parse_args()
data_path = args.data_path

TRAIN_PERCENT = 0.85
VALIDATION_PERCENT = 0.15

occurences_count = dict()
for file in os.listdir(data_path):
    stem = file.split(".")[0]
    suffix = "." + file.split(".")[1]

    if stem not in occurences_count:
        count = {}
        count[suffix] = 1
        occurences_count[stem] = count
    else:
        occurences_count[stem][suffix] = occurences_count[stem][suffix] =+ 1

# map to array only containing valid pairs
valid_pairs = []
for key, value in occurences_count.items():
    try:
        if value[".gui"] == 1 and value[".png"] == 1:
            valid_pairs.append(key)
    except:
        print(f'File {key} is not a valid pair')



number_of_examples = len(valid_pairs)
print(f'Found a total of {number_of_examples} valid examples')

train_split = floor(number_of_examples * TRAIN_PERCENT)
test_split = floor(number_of_examples * VALIDATION_PERCENT)

train_set = valid_pairs[:train_split]
test_set = valid_pairs[train_split:]

dataset_splits = {"train": train_set, "test": test_set}

for key, value in dataset_splits.items():
    filepath = os.path.join(os.path.dirname(data_path), f'{key}_dataset.txt')

    with open(filepath, "w") as writer:
        for example in value:
            writer.write(example + "\n")
