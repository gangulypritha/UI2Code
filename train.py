import argparse
from vocab import Vocab
import torch
from torch.utils.data import DataLoader
from dataset import UI2CodeDataset
from utils import collate_fn, save_model, resnet_img_transformation
from models import Encoder, Decoder
import math
import os

parser = argparse.ArgumentParser(description='Train the model')

parser.add_argument("--data_path", type=str, help="Path to the dataset")
parser.add_argument("--cuda", action='store_true',
                    default=True, help="Use cuda or not")
parser.add_argument("--img_crop_size", type=int, default=224)
parser.add_argument("--split", type=str, default="train")
parser.add_argument("--save_after_epochs", type=int, default=1,
                    help="Save model checkpoint every n epochs")
parser.add_argument("--models_dir", type=str, help="The dir where the trained models are saved")
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--epochs", type=int, default=15)
parser.add_argument("--lr", type=float, default=1e-3,
                    help="Learning Rate")
parser.add_argument("--print_freq", type=int, default=1,
                    help="Print training stats every n epochs")
parser.add_argument("--seed", type=int, default=2020,
                    help="The random seed for reproducing")

args = parser.parse_args()
vocab_file_path = os.path.join(os.path.dirname(args.data_path), "vocab.txt")

print("Training args:", args)

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# Load the vocab file
vocab = Vocab(vocab_file_path)
assert len(vocab) > 0

# Setup GPU
use_cuda = True if args.cuda and torch.cuda.is_available() else False
device = torch.device("cuda" if use_cuda else "cpu")

transform_imgs = resnet_img_transformation(args.img_crop_size)

# Creating the data loader
train_loader = DataLoader(
    UI2CodeDataset(args.data_path, args.split,
                    vocab, transform=transform_imgs),
    batch_size=args.batch_size,
    collate_fn=lambda data: collate_fn(data, vocab=vocab),
    pin_memory=True if use_cuda else False,
    num_workers=4,
    drop_last=True)
print("Created data loader")

# Creating the models
embed_size = 256
hidden_size = 512
num_layers = 1
lr = args.lr

encoder = Encoder(embed_size)
decoder = Decoder(embed_size, hidden_size, len(vocab), num_layers)

encoder = encoder.to(device)
decoder = decoder.to(device)

# Define optimizer and loss function
criterion = torch.nn.CrossEntropyLoss()
params = list(decoder.parameters()) + list(encoder.linear.parameters()
                                           ) + list(encoder.BatchNorm.parameters())
optimizer = torch.optim.Adam(params, lr=lr)

# Training the model
for epoch in range(args.epochs):
    for i, (images, captions, lengths) in enumerate(train_loader):
        images = images.to(device)
        captions = captions.to(device)

        targets = torch.nn.utils.rnn.pack_padded_sequence(
            input=captions, lengths=lengths, batch_first=True)[0]

        encoder.zero_grad()
        decoder.zero_grad()

        features = encoder(images)
        output = decoder(features, captions, lengths)
        loss = criterion(output, targets)

        loss.backward()
        optimizer.step()

        loss = loss.item()

        if epoch % args.print_freq == 0 and i == 0:
            print(
                f'Epoch : {epoch} || Loss : {loss:.4f} || Perplexity : {math.exp(loss):.4f}')

        if epoch != 0 and epoch % args.save_after_epochs == 0 and i % len(train_loader) == 0:
            save_model(args.models_dir, encoder, decoder,
                       optimizer, epoch, loss, args.batch_size, vocab)
            print("Saved model checkpoint")

print("Done Training!")

save_model(args.models_dir, encoder, decoder,
           optimizer, epoch, loss, args.batch_size, vocab)
print("Saved final model")
