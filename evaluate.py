import argparse
from vocab import Vocab
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import UI2CodeDataset
from utils import collate_fn, save_model, ids_to_tokens, generate_visualization_object, resnet_img_transformation
from models import Encoder, Decoder
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import math
from tqdm import tqdm
import os

parser = argparse.ArgumentParser(description='Evaluate the model')

parser.add_argument("--model_file_path", type=str,
                    help="Path to the trained model file", required=True)
parser.add_argument("--data_path", type=str, help="Datapath")
parser.add_argument("--cuda", action='store_true',
                    default=True, help="Use cuda or not")
parser.add_argument("--img_crop_size", type=int, default=224)
parser.add_argument("--embed_size", type=int, default=256)
parser.add_argument("--hidden_size", type=int, default=512)
parser.add_argument("--num_layers", type=int, default=1)
parser.add_argument("--split", type=str, default="validation")
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--seed", type=int, default=2020,
                    help="The random seed for reproducing ")

args = parser.parse_args()
print("Evaluation args:", args)

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# Setup GPU
use_cuda = True if args.cuda and torch.cuda.is_available() else False
device = torch.device("cuda" if use_cuda else "cpu")

# Loading the model
embed_size = args.embed_size
hidden_size = args.hidden_size
num_layers = args.num_layers

assert os.path.exists(args.model_file_path)
loaded_model = torch.load(args.model_file_path)

vocab = loaded_model["vocab"]

encoder = Encoder(embed_size)
decoder = Decoder(embed_size, hidden_size, len(vocab), num_layers)

encoder.load_state_dict(loaded_model["encoder_model_state_dict"])
decoder.load_state_dict(loaded_model["decoder_model_state_dict"])

encoder = encoder.to(device)
decoder = decoder.to(device)

transform_imgs = resnet_img_transformation(args.img_crop_size)

# Creating the data loader
data_loader = DataLoader(
    UI2CodeDataset(args.data_path, args.split,
                    vocab, transform=transform_imgs),
    batch_size=args.batch_size,
    collate_fn=collate_fn,
    pin_memory=True if use_cuda else False,
    num_workers=4,
    drop_last=True)

# Evaluate the model
encoder.eval()
decoder.eval()

predictions = []
targets = []
for i, (image, caption) in enumerate(tqdm(data_loader.dataset)):
    image = image.to(device)
    caption = caption.to(device)

    features = encoder(image.unsqueeze(0))

    sample_ids = decoder.sample(features)
    sample_ids = sample_ids.cpu().data.numpy()

    predictions.append(sample_ids)
    targets.append(caption.cpu().numpy())

predictions = [ids_to_tokens(vocab, prediction) for prediction in predictions]
targets = [ids_to_tokens(vocab, target) for target in targets]

bleu = corpus_bleu([[target] for target in targets], predictions,
                   smoothing_function=SmoothingFunction().method4)
print("BLEU score: {}".format(bleu))

generate_visualization_object(data_loader.dataset, predictions, targets)
print("Predictions, targets combined into a pkl object! Use the visualisation notebook for insights!")
