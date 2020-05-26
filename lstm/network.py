from torch import nn
import torch

# -*- coding: utf-8 -*-
class Network(nn.Module):

    def __init__(self, input_size, hidden_units, layers_num, dropout_prob=0):
        # Call the parent init function (required!)
        super().__init__()
        # Define recurrent layer
        self.rnn = nn.LSTM(input_size=input_size,
                           hidden_size=hidden_units,
                           num_layers=layers_num,
                           dropout=dropout_prob,
                           batch_first=True)
        # Define output layer
        self.out = nn.Linear(hidden_units, input_size)

    def forward(self, x, state=None):
        # LSTM
        x, rnn_state = self.rnn(x, state)
        # Linear layer
        x = self.out(x)
        return x, rnn_state

def train_batch(net, batch_onehot, loss_fn, optimizer):

    batch_onehot = batch_onehot.float()
    ### Prepare network input and labels
    # Get the labels (the last word of each sequence)
    labels_onehot = batch_onehot[:, -1, :]
    # Remove the labels from the input tensor
    net_input = batch_onehot[:, :-1, :]

    ### Forward pass
    # Eventually clear previous recorded gradients
    optimizer.zero_grad()
    # Forward pass
    net_out, _ = net(net_input)
    ### Update network

    # Evaluate loss only for last output
    loss = loss_fn(net_out[:, -1, :], labels_onehot)
    # Backward pass
    loss.backward()
    # Update
    optimizer.step()
    # Return average batch loss
    return float(loss.data), net_out, labels_onehot
