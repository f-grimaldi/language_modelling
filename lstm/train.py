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
parser.add_argument('--layers_num',    type=int,   default=2,                           help='Number of RNN stacked layers')
parser.add_argument('--dropout_prob',  type=float, default=0.3,                         help='Dropout probability')

# Training
parser.add_argument('--batchnumber',   type=int,   default=80,                        help='Number of batch')
parser.add_argument('--num_epochs',    type=int,   default=400,                         help='Number of training epochs')
parser.add_argument('--verbose',       type=int,   default=5,                         help='Verbose')

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

    ### Test dataloader
    #alphabet_len = len(dataset.alphabet)
    trans = transforms.Compose([ds.RandomCrop(crop_len),
                                ds.OneHotEncoder(),
                                ds.ToTensor()
                                ])
    dataset = ds.LOTRDataset(text, emb, transform=trans)
    dataloader = DataLoader(dataset, batch_size=len(dataset.chapter_list)//args.batchnumber, shuffle=True)

    ### Set cuda
    device = torch.device('cuda')

    ### Get validation
    for batch_sample in dataloader:
        batch_onehot = batch_sample['encoded_onehot']
    validation_batch = batch_onehot.float().to(device)

    ### Initialize Network
    #Define Network
    net = network.Network(input_size=emb.shape[1],
                          hidden_units=args.hidden_units,
                          layers_num=args.layers_num,
                          dropout_prob=args.dropout_prob)

    #Move Network into GPU
    net = net.to(device)
    # Define optimizer
    optimizer = optim.Adam(net.parameters(), lr = 0.003)
    # Define loss function
    loss_fn = nn.MSELoss()

    ### Init Training
    train_loss, validation_loss = [], []
    verbose = args.verbose

    # Start training
    train_loss, val_loss = [], []
    # Start training
    for epoch in range(args.num_epochs):
        start = time.time()
        if epoch%verbose == 0:
            print('\n##################################')
            print('## EPOCH %d' % (epoch + 1))
            print('##################################')

        b_losses = []
        # Iterate batches
        for batch_sample in dataloader:
            # Extract batch
            batch_onehot = batch_sample['encoded_onehot'].to(device)
            if batch_onehot.shape[0] != validation_batch.shape[0]:
              # Update network
              batch_loss, out, y_true = network.train_batch(net, batch_onehot, loss_fn, optimizer)
              b_losses.append(batch_loss)


            with torch.no_grad():
                y_validation = validation_batch[:, -1, :]
                # Remove the labels from the input tensor
                val_input = validation_batch[:, :-1, :]
                validation_pred, _ = net(val_input)
                ### Update network
                # Evaluate loss only for last output
                loss_val = loss_fn(validation_pred[:, -1, :], y_validation)
                val_loss.append(loss_val)


        train_loss.append(torch.mean(torch.tensor(b_losses)))
        print('Avarage loss: {}'.format(torch.mean(torch.tensor(b_losses))), end='\t')
        print('Validation loss: {}'.format(loss_val.data), end='\t')
        print('Time: {}'.format(np.round(time.time()-start, 3)))


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
    with open(out_dir / 'word_to_vec.json', 'w') as f:
        for i in dataset.char_to_number.keys():
            dataset.char_to_number[i] = dataset.char_to_number[i].tolist() 
        json.dump(dataset.char_to_number, f, indent=4)
    emb.to_csv(r'{}\\embedding_matrix.csv'.format(out_dir))
