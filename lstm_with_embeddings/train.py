import re
import torch
import argparse
import json
import time

import network
import dataset as ds

import numpy as np
import pandas as pd

from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from functools import reduce
from torchvision import transforms
from torch import optim, nn



##############################
##############################
## PARAMETERS
##############################
parser = argparse.ArgumentParser(description='Train the sonnet generator network.')

# Dataset
parser.add_argument('--datasetpath',   type=str,  default='lotr.txt',                   help='Path of the train txt file')
parser.add_argument('--embeddingpath', type=str,  default=r'glove.6B\glove.6B.50d.txt', help='Path of the embedding')
parser.add_argument('--crop_len',      type=int,  default=7,                            help='Number of input letters')

# Network
parser.add_argument('--hidden_units',  type=int,   default=256,                         help='Number of RNN hidden units')
parser.add_argument('--layers_num',    type=int,   default=4,                           help='Number of RNN stacked layers')
parser.add_argument('--dropout_prob',  type=float, default=0,                         help='Dropout probability')

# Training
parser.add_argument('--batchnumber',   type=int,   default=90,                        help='Number of batch')
parser.add_argument('--num_epochs',    type=int,   default=2000,                         help='Number of training epochs')
parser.add_argument('--verbose',       type=int,   default=1,                         help='Verbose')

# Save
parser.add_argument('--out_dir',       type=str,   default='model',                     help='Path of models and params')

##############################
##############################
##############################

if __name__ == '__main__':

    ### Parse input arguments
    args = parser.parse_args()

    ### Check device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print('Selected device:', device)
    ###Set arguments
    crop_len = args.crop_len

    ###Set paths
    text_path = args.datasetpath
    embedding_path = args.embeddingpath

    ###Load text and embeddings
    text = ds.clean_text(text_path)
    emb = ds.embedding_matrix(embedding_path, text, normalize=True)
    emb = ds.clean_embedding(emb, text, min_freq=5)

    ### Test dataloader
    #alphabet_len = len(dataset.alphabet)
    trans = transforms.Compose([ds.RandomCrop(crop_len),
                                ds.OneHotEncoder(emb),
                                ds.ToTensor()
                                ])
    dataset = ds.LOTRDataset(text, emb, transform=trans)
    dataloader = DataLoader(dataset, batch_size=len(dataset.chapter_list)//args.batchnumber, shuffle=True)

    ### Set cuda
    device = torch.device('cuda')

    ### Get validation
    for batch_sample in dataloader:
        batch_onehot = batch_sample['encoded_onehot']
        print(batch_onehot.shape)
    validation_batch = batch_onehot.to(device)

    ### Initialize Network
    #Define Network
    net = network.Network(embedding_matrix = emb.values,
                          hidden_units=args.hidden_units,
                          layers_num=args.layers_num,
                          dropout_prob=args.dropout_prob)

    #Move Network into GPU
    net = net.to(device)
    # Define optimizer
    optimizer = optim.Adam(net.parameters(), lr = 0.003)
    # Define loss function
    loss_fn = nn.CrossEntropyLoss()

    ### Init Training
    train_loss, validation_loss = [], []
    verbose = args.verbose

    # Start training
    for epoch in range(args.num_epochs):

        #Start time
        start = time.time()

        #Print state
        if epoch%verbose == 0:
            print('## EPOCH %d' % (epoch + 1), end = '\t')

        batch_losses = []
        for batch_sample in dataloader:
        # Extract batch
            if batch_sample['encoded_onehot'].shape[0] != validation_batch.shape[0]:
              net.train()
              batch_onehot = batch_sample['encoded_onehot'].to(device).long()
              batch_onehot = torch.argmax(batch_onehot, dim = 2)

              batch_loss, out, y_true = network.train_batch(net, batch_onehot, loss_fn, optimizer)
              batch_losses.append(batch_loss)

            else:#
              with torch.no_grad():
                ###TARGET (51), OUT (51, 5092)
                net.eval()
                validation_inputs = torch.argmax(validation_batch, dim = 2).long()
                net_out, _ = net(validation_inputs)
                loss_vn = loss_fn(net_out[:, -1, :], validation_inputs[:, -1])

        print('Loss: {}'.format(np.round(float(torch.mean(torch.tensor(batch_losses))), 5)), end='\t')
        print('Loss: {}'.format(np.round(float(torch.mean(loss_vn)), 5)), end='\t')
        print('Time: {}'.format(np.round(time.time()-start, 2)))

        train_loss.append(np.round(float(torch.mean(torch.tensor(batch_losses))), 5))
        validation_loss.append(np.round(float(torch.mean(loss_vn)), 5))

    ### Save all needed parameters
    # Create output dir
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Save network parameters
    torch.save(net.state_dict(), out_dir / 'net_params.pth')
    # Save training parameters
    with open(out_dir / 'training_args.json', 'w') as f:
        json.dump(vars(args), f, indent=4)
    # Save encoder dictionary
    with open(out_dir / 'word_to_number.json', 'w') as f:
        json.dump(dataset.word_to_number, f, indent=4)
    with open(out_dir / 'number_to_word.json', 'w') as f:
        json.dump(dataset.number_to_word, f, indent=4)
    emb.to_csv(r'{}\\embedding_matrix.csv'.format(out_dir))
