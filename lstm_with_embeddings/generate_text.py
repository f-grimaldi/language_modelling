import numpy as np
import pandas as pd

import json
import torch
import argparse

from torch import nn
from pathlib import Path


import network
import dataset as ds


##############################
##############################
## PARAMETERS
##############################
parser = argparse.ArgumentParser(description='Generate sonnet starting from a given text')

parser.add_argument('--datasetpath',    type=str,   default='lotr.txt',                       help='Path of the train txt file')
parser.add_argument('--seed',           type=str,   default='he went up and down the shire',  help='Initial text of the sonnet')
parser.add_argument('--model_dir',      type=str,   default='pre_trained_model',              help='Network model directory')
parser.add_argument('--length_text',    type=int,   default=200,                              help='Length (words) of text to generate')


if __name__ == '__main__':

    ### Parse input arguments
    args = parser.parse_args()

    #device = torch.device("cpu")
    #%% Load embedding matrix
    text_path = args.datasetpath

    ###Loading and preparing the embedding matrix
    #%% Load training parameters
    model_dir = Path(args.model_dir)
    emb = pd.read_csv(r'{}\\embedding_matrix.csv'.format(model_dir))
    emb.index = emb.iloc[:, 0]
    emb = emb.iloc[:, 1:]
    print('Loading model from: %s' % model_dir)
    print()
    training_args = json.load(open(model_dir / 'training_args.json'))

    #%% Load encoder and decoder dictionaries
    encoder = json.load(open(model_dir / 'word_to_number.json'))
    decoder = json.load(open(model_dir / 'number_to_word.json'))

    #%% Initialize network
    net = network.Network(embedding_matrix=emb.values,
                          hidden_units=training_args['hidden_units'],
                          layers_num=training_args['layers_num'])

    #%% Load network trained parameters
    net.load_state_dict(torch.load(model_dir / 'net_params.pth', map_location='cpu'))

    net.eval() # Evaluation mode (e.g. disable dropout)

    state = args.seed
    with torch.no_grad():
        # Encode seed
        seed_encoded = ds.encode_text(encoder, state.lower())
        # One hot matrix
        # To tensor
        seed_onehot = torch.tensor(seed_encoded).long()
        #print(seed_onehot)
        # Add batch axis
        seed_onehot = seed_onehot.unsqueeze(0)
        # Forward pass
                                                                                #seed_onehot = seed_onehot.to(device)
        net_out, net_state = net(seed_onehot)
        # Get the most probable last output index
        next_word_encoded = net_out[:, -1, :]
        encoded = [int(torch.argmax(c)) for c in next_word_encoded]
        next_word = [decoder[str(c)] for c in encoded]
                                                                                #next_word = ds.decode_text(decoder, next_word_encoded.cpu().numpy())
        state += ' ' + next_word[0]
        # Print the seed letters
        #print(state, end=' ', flush=True)
    #%% Generate sonnet
    new_line_count = 0
    tot_char_count = 0
    while True:
        with torch.no_grad(): # No need to track the gradients
            # The new network input is the one hot encoding of the last chosen letter
            if len(state.split()) > 5:
              state_enc = ' '.join(state.split()[-10:])
              seed_encoded = ds.encode_text(encoder, state_enc.lower())
            else:
              seed_encoded = ds.encode_text(encoder, state.lower())

            seed_onehot = torch.tensor(seed_encoded).long()
            # Add batch axis
            seed_onehot = seed_onehot.unsqueeze(0)
            #print(net_input.shape)
            #net_input = net_input.unsqueeze(0)
            # Forward pass
            net_out, net_state = net(seed_onehot, net_state)

            # Get the most probable letter index
            next_word_encoded = net_out[:, -1, :]
            encoded = [int(torch.argmax(c)) for c in next_word_encoded]
            next_word = [decoder[str(c)] for c in encoded]
                                                                                        #next_word = ds.decode_text(decoder, next_word_encoded.cpu().numpy())
            state += ' ' + next_word[0]
            # Print the seed letters
            # Count total letters
            tot_char_count += 1
            # Count new lines
            # Break if 14 lines or 2000 letters
            if tot_char_count > args.length_text:
                break

    print('TEXT:')
    after_point = False
    for i in state.split():
        if i in set(('.', '?', '!')):
          after_point = True

        if i in set(('.', ',', ';', ':', '?', '!')):
          print(i, end='')
        else:
          if after_point:
            print(' {}'.format(i.capitalize()), end='')
            after_point=False
          else:
            print(' {}'.format(i), end='')
    print()
