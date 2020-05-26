from torch import nn
import torch


def create_emb_layer(weights_matrix, non_trainable=False):
    weights_matrix = torch.from_numpy(weights_matrix)
    inp_size, out_size = weights_matrix.size()

    emb_layer = nn.Embedding(inp_size, out_size)
    emb_layer.load_state_dict({'weight': weights_matrix})

    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, inp_size, out_size


class Network(nn.Module):

    def __init__(self, embedding_matrix, hidden_units, layers_num, dropout_prob=0):
        # Call the parent init function (required!)
        super().__init__()
        # Define recurrent layer
        self.embedding, num_embeddings, embedding_dim = create_emb_layer(embedding_matrix, True)
        self.rnn = nn.LSTM(input_size=embedding_dim,
                           hidden_size=hidden_units,
                           num_layers=layers_num,
                           dropout=dropout_prob,
                           batch_first=True)
        self.out = nn.Linear(hidden_units, num_embeddings)
        # Define output layer
        #self.act = nn.functional.softmax

    def forward(self, x, state=None):
        # LSTM
        x = self.embedding(x)
        x, rnn_state = self.rnn(x, state)
        # Linear layer
        x = self.out(x)
        return x, rnn_state


def train_batch(net, batch_onehot, loss_fn, optimizer):

    batch_onehot = batch_onehot.long()

    ### Prepare network input and labels
    # Get the labels (the last word of each sequence)
    labels_onehot = batch_onehot[:, -1]
    # Remove the labels from the input tensor
    net_input = batch_onehot[:, :-1]

    ### Forward pass
    # Eventually clear previous recorded gradients
    optimizer.zero_grad()
    # Forward pass
    net_out, _ = net(net_input)
    ### Update network
    # Evaluate loss only for last output
    loss = loss_fn(net_out[:, -1, :], labels_onehot)
    # Backward pass
    #loss.backward(retain_graph=True)
    loss.backward()
    # Update
    optimizer.step()

    # Return average batch loss
    return float(loss.data), net_out, labels_onehot
