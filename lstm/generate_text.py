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

parser.add_argument('--seed', type=str, default='the shire was', help='Initial text of the sonnet')
parser.add_argument('--model_dir',   type=str, default='pre_trained_model', help='Network model directory')
parser.add_argument('--length_text',   type=int, default=200, help='Length (words) of text to generate')

if __name__ == '__main__':

    ### Parse input arguments
    args = parser.parse_args()

    ###Loading and preparing the embedding matrix
    #%% Load training parameters
    model_dir = Path(args.model_dir)

    ### Load Embedding
    emb = pd.read_csv(r'{}\\embedding_matrix.csv'.format(model_dir))
    emb.index = emb.iloc[:, 0]
    emb = emb.iloc[:, 1:]

    print('Loading model from: %s' % model_dir)
    print()
    training_args = json.load(open(model_dir / 'training_args.json'))

    #%% Load encoder dictionaries
    encoder = json.load(open(model_dir / 'word_to_vec.json'))

    #%% Initialize network
    net = network.Network(input_size=emb.shape[1],
                          hidden_units=training_args['hidden_units'],
                          layers_num=training_args['layers_num'],
                          dropout_prob=training_args['dropout_prob'])

    #%% Load network trained parameters
    net.load_state_dict(torch.load(model_dir / 'net_params.pth', map_location='cpu'))

    net.eval() # Evaluation mode (e.g. disable dropout)

    state = args.seed
    X = emb.values

    with torch.no_grad():
        # Encode seed
        seed_encoded = ds.encode_text(encoder, state.lower(), emb)
        # One hot matrix
        #seed_onehot = create_one_hot_matrix(seed_encoded, 47)
        # To tensor
        seed_onehot = torch.tensor(seed_encoded).float()
        #print(seed_onehot)
        # Add batch axis
        seed_onehot = seed_onehot.unsqueeze(0)
        # Forward pass
        #seed_onehot = seed_onehot.to(device)
        net_out, net_state = net(seed_onehot)
        # Get the most probable last output index
        next_word_encoded = net_out[:, -1, :]
        closest_value = np.linalg.norm((X - next_word_encoded.to('cpu').numpy()[0]), axis = 1)
        closest_word = emb.index[np.argmin(closest_value)]
        state += ' ' + closest_word
        # Print the seed letters
        #print(state, end=' ', flush=True)
    #%% Generate sonnet
    new_line_count = 0
    tot_char_count = 0
    while True:
        with torch.no_grad(): # No need to track the gradients
            # The new network input is the one hot encoding of the last chosen letter
            net_input = ds.encode_text(encoder, state.lower(), emb)
            net_input = torch.reshape(torch.tensor(net_input).float(), (1, -1, 50))
            #print(net_input.shape)
            #net_input = net_input.unsqueeze(0)
            # Forward pass
            #net_input = net_input.to(device)
            net_out, net_state = net(net_input, net_state)

            # Get the most probable letter index
            closest_value = np.linalg.norm((X - net_out[:, -1, :].to('cpu').numpy()[0]), axis = 1)
            closest_word = emb.index[np.argmin(closest_value)]
            state += ' ' + closest_word
            #print(closest_word, end=' ', flush=True)
            # Count total letters
            tot_char_count += 1
            # Count new lines
            # Break if 14 lines or 2000 letters
            if tot_char_count > args.length_text:
                break

    print('TEXT:')
    after_point = False
    previous = None
    for i in state.split():
        if i == previous:
            continue
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
        previous = i
    print()
