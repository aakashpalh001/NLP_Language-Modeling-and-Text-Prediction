from typing import Iterator

import torch
import torch.nn as nn
from torch.nn import Parameter


# Here is a pseudocode to help with your LSTM implementation. 
# You can add new methods and/or change the signature (i.e., the input parameters) of the methods.
class LSTM(nn.Module):
    def __init__(self, input_size, n_characters, hidden_size, output_size, n_layers=1):
        """Think about which (hyper-)parameters your model needs; i.e., parameters that determine the
        exact shape (as opposed to the architecture) of the model. There's an embedding layer, which needs
        to know how many elements it needs to embed, and into vectors of what size. There's a recurrent layer,
        which needs to know the size of its input (coming from the embedding layer). PyTorch also makes
        it easy to create a stack of such layers in one command; the size of the stack can be given
        here. Finally, the output of the recurrent layer(s) needs to be projected again into a vector
        of a specified size."""
        ############################ STUDENT SOLUTION ############################
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.n_characters = n_characters
        self.n_layers = n_layers

        self.embedding = nn.Embedding(n_characters, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, n_characters)
        # Initialize hidden and cell states
        self.hidden_state = None
        self.cell_state = None


        #self.lstm = nn.LSTM(n_characters, hidden_size, n_layers, batch_first=True)
        #self.fc = nn.Linear(hidden_size, n_characters)
        ##########################################################################

    def forward(self, x, hidden):
        """Your implementation should accept input character, hidden and cell state,
        and output the next character distribution and the updated hidden and cell state."""
        ############################ STUDENT SOLUTION ############################
        # YOUR CODE HERE
        x = self.embedding(x)
        output, hidden = self.lstm(x, hidden)
        output = self.fc(output)
        return output, hidden

        ##########################################################################


    def init_hidden(self, batch_size = 1):
        """Finally, you need to initialize the (actual) parameters of the model (the weight
        tensors) with the correct shapes."""
        ############################ STUDENT SOLUTION ############################
        # YOUR CODE HERE
        hidden_state = torch.zeros(self.n_layers, batch_size, self.hidden_size)
        cell_state = torch.zeros(self.n_layers, batch_size, self.hidden_size)
        return hidden_state, cell_state
        ##########################################################################

