import argparse
import json
import torch
import network
import data
import numpy as np


####################
# Setting arguments
####################
parser = argparse.ArgumentParser(description='PyTorc Text generation Model, based on Lord of The Rings, The Hobbit and The Silmarrillon')

# Model parameters.
parser.add_argument('--data',         type=str,    default='./data',             help='location of the data corpus')
parser.add_argument('--n',            type=int,    default='200',                help='number of words to generate')
parser.add_argument('--seed',         type=str,    default='the',                help='initial seed')
parser.add_argument('--random_seed',  type=int,    default= 2,                   help='set initial random seed')
parser.add_argument('--model_dir',    type=str,    default='pre_trained_model',  help='where to fetch the model')

args = parser.parse_args()

####################
# Setting parameters
####################
# Set the random seed manually for reproducibility.
torch.manual_seed(args.random_seed)

# Set the Device
device = torch.device("cuda") if torch.cuda.is_available() else torch.cpu()
print(device)

# Set dataset arguments
corpus = data.Corpus(args.data)
ntokens = len(corpus.dictionary)

####################
# Retrieve the model
####################
# Retrieve training arguments
training_args = json.load(open('{}//training_args.json'.format(args.model_dir)))

# Build the model
model = network.TransformerModel(ntokens,
                                 training_args['emsize'],
                                 training_args['nhead'],
                                 training_args['nhid'],
                                 training_args['nlayers'],
                                 training_args['dropout']).to(device)

# Update the weights
model.load_state_dict(torch.load('{}//net_params.pth'.format(args.model_dir), map_location='cpu'))

# Set the model in evaluation state
model.eval()

####################
# Generating the text
####################
# Set initial word
state = args.seed
word = args.seed.split()[-1]

#Handle the situation where the seed is not contained in the dictionary
if word in corpus.dictionary.word2idx:
    input = torch.tensor(np.reshape(corpus.dictionary.word2idx[word], (1, 1))).long().to(device)
else:
    print('No such words into the dictionary. Starting with random seed')
    input = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)
    print('Random seed: {}'.format(corpus.dictionary.idx2word[input]))
    print('\n{}'.format(corpus.dictionary.idx2word[input]), end=' ')

#Generate next word
with torch.no_grad():  # no tracking history
    for i in range(args.n):
        # Get output
        output = model(input, False)
        word_weights = output[-1].squeeze().exp().cpu()

        # Sample word from output distribution
        word_idx = torch.multinomial(word_weights, 1)[0]
        word_tensor = torch.Tensor([[word_idx]]).long().to(device)

        # Concatenate the word predicted with the current state
        input = torch.cat([input, word_tensor], 0)
        word = corpus.dictionary.idx2word[word_idx]
        if word == '<eos>':
            continue
        state = '{} {}'.format(state, word)

####################
# Print the text
####################

# Set punktuations signs and upper case signs
punck = ['!', '?', '.', ';', ':', ',',"'"]
upcase = ['?',  '!',  '.']

# Set initial params
after_point = False
new_line_counter = 0
previous = '_'

# Print initial state
print('TEXT:')
print('{}'.format(state.split()[0]), end = '')

# Print next word following some given rules
for i in state.split()[1:]:
    #If it's the same word try again
    if i == previous:
        continue

    #Update previou word
    previous = i

    #Increment
    new_line_counter += 1

    #Signal the next letter must start in uppercase
    if i in upcase:
      after_point = True

    #Signal there is a full stop and we start new_line
    if i == '.' and new_line_counter > 10:
      new_line_counter = 0
      print('.')

    #Signal there is a punktuation sign so we don't add anywhite space
    elif i in punck:
      print(i, end='')
      new_line_counter -= 1

    #If there isn't any special char we add the word and the whitespace
    else:
      if after_point:
        if new_line_counter > 1:
            print(' {}'.format(i.capitalize()), end='')
            after_point=False
        #If it's a new line we don't add the white space
        else:
            print('{}'.format(i.capitalize()), end='')
            after_point=False
      else:
        print(' {}'.format(i), end='')
