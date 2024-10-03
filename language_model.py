import torch
import torch.nn as nn
import string
import time
import unidecode
import matplotlib.pyplot as plt

from utils import char_tensor, random_training_set, time_since, random_chunk, CHUNK_LEN
from evaluation import compute_bpc
from model.model import LSTM
from datetime import datetime



def generate(decoder, prime_str='A', predict_len=100, temperature=0.8):

    print(f"Generating text... Prime: '{prime_str}', Length: {predict_len}, Temp: {temperature}")

    hidden, cell = decoder.init_hidden()
    prime_input = char_tensor(prime_str)
    predicted = prime_str
    all_characters = string.printable

    # Use priming string to "build up" hidden state
    for p in range(len(prime_str) - 1):
        _, (hidden, cell) = decoder(prime_input[p], (hidden, cell))
    inp = prime_input[-1]

    for p in range(predict_len):
        output, (hidden, cell) = decoder(inp, (hidden, cell))

        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]

        # Add predicted character to string and use as next input
        predicted_char = all_characters[top_i]
        predicted += predicted_char
        inp = char_tensor(predicted_char)

    print(f"Generated text: {predicted}")

    return predicted


def train_step(decoder, decoder_optimizer, inp, target):
    hidden, cell = decoder.init_hidden()
    decoder.zero_grad()
    loss = 0
    criterion = nn.CrossEntropyLoss()

    for c in range(CHUNK_LEN):
        inp_batch = inp[c].unsqueeze(0).unsqueeze(0)
        target_batch = target[c].unsqueeze(0)

        output, (hidden, cell) = decoder(inp_batch, (hidden, cell))
        loss += criterion(output.view(-1,decoder.n_characters), target_batch)
        print(f"Chunk {c+1}/{CHUNK_LEN}, Loss: {loss.item() / CHUNK_LEN}")


    loss.backward()
    decoder_optimizer.step()


    return loss.item() / CHUNK_LEN


def tuner(n_epochs=250, print_every=100, plot_every=20, hidden_size=128, n_layers=2,
          lr=0.005, start_string='A', prediction_length=100, temperature=0.8):
    print(f"Starting training... LR: {lr}, Hidden Size: {hidden_size}, Layers: {n_layers}")

    losses = []
    decoder = LSTM(
        input_size=len(string.printable),
        n_characters=len(string.printable),
        hidden_size=64,

        #hidden_size=hidden_size,
        output_size=len(string.printable),
        n_layers=1)

    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    start = time.time()
    for epoch in range(1, n_epochs + 1):
        decoder_optimizer.zero_grad()
        inp, target = random_training_set()
        loss = train_step(decoder, decoder_optimizer, inp, target)
        losses.append(loss)

        if epoch % print_every == 0:
            elapsed_time = time_since(start)
            print(f"Epoch {epoch}/{n_epochs} - Loss: {loss:.4f} - Elapsed Time: {elapsed_time}")


            # print(
            #     f"Learning Rate {lr} | Epoch {epoch} ({time_since(start)}): Loss {loss:.4f}")


    print("Losses:", losses)

    epochs_range = list(range(1, len(losses) * plot_every + 1, plot_every))
    print("Epochs range:", epochs_range)
    plt.plot(epochs_range, losses, label=f"LR: {lr}")

    #plt.plot(losses, list(range(1, len(losses) * plot_every + 1, plot_every)), label=f"LR: {lr}")

    return decoder  # Return the trained model

        # YOUR CODE HERE
        #     TODO:
        #         1) Implement a `tuner` that wraps over the training process (i.e. part
        #            of code that is ran with `default_train` flag) where you can
        #            adjust the hyperparameters
        #         2) This tuner will be used for `custom_train`, `plot_loss`, and
        #            `diff_temp` functions, so it should also accomodate function needed by
        #            those function (e.g. returning trained model to compute BPC and
        #            losses for plotting purpose).

        ################################### STUDENT SOLUTION #######################

        ############################################################################

def plot_loss(lr_list):
    print("Plotting training losses for learning rates:", lr_list)
    for lr in [0.005, 0.0005]:
        _ = tuner(lr=lr)

    plt.xlabel('Epochs')
    plt.xlim(0, 3000)
    plt.ylabel('Loss')
    plt.title('Training Loss for Different Learning Rates')
    plt.legend()  # Show legend with labels for each learning rate
    filename = rf"Plots\{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"
    plt.savefig(filename)
    plt.show()  # Display the combined plot


    #     TODO:
    #         1) Using `tuner()` function, train X models where X is len(lr_list),
    #         and plot the training loss of each model on the same graph.
    #         2) Don't forget to add an entry for each experiment to the legend of the graph.
    #         Each graph should contain no more than 10 experiments.
    ###################################### STUDENT SOLUTION ##########################
    ##################################################################################

def diff_temp(temp_list):
    print("Generating text at different temperatures...")
    TEST_PATH = './data/dickens_test.txt'
    for temp in temp_list:
        print(f'Temperature: {temp}')
        chunk = random_chunk()
        prime_str = chunk[:10]
        model, _ = tuner()
        print(generate(model, prime_str, 200, temp), '\n')

    #     TODO:
    #         1) Using `tuner()` function, try to generate strings by using different temperature
    #         from `temp_list`.
    #         2) In order to do this, create chunks from the test set (with 200 characters length)
    #         and take first 10 characters of a randomly chosen chunk as a priming string.
    #         3) What happen with the output when you increase or decrease the temperature?
    ################################ STUDENT SOLUTION ################################
    ##################################################################################

def custom_train(hyperparam_list):


    """
    Train model with X different set of hyperparameters, where X is 
    len(hyperparam_list).

    Args:
        hyperparam_list: list of dict of hyperparameter settings

    Returns:
        bpc_dict: dict of bpc score for each set of hyperparameters.
    """
    #string = unidecode.unidecode(open(TEST_PATH, 'r').read())
    TEST_PATH = './data/dickens_test.txt'
    string = unidecode.unidecode(open(TEST_PATH, 'r').read())
    #         1) Using `tuner()` function, train X models with different
    # set of hyperparameters and compute their BPC scores on the test set.
    ################################# STUDENT SOLUTION ##########################
    bpc_dict = {}
    test_text = unidecode.unidecode(open('./data/dickens_test.txt', 'r').read())
    print("Training models with different hyperparameters...")

    for params in hyperparam_list:
        decoder = tuner(lr=params['lr'], hidden_size=params['hidden_size'])
        generated_text = generate(decoder)
        bpc = compute_bpc(decoder, test_text)
        bpc_dict[str(params)] = bpc
        print(f"Params: {params}, BPC: {bpc}")


    return bpc_dict




    #     TODO:
    #         1) Using `tuner()` function, train X models with different
    #         set of hyperparameters and compute their BPC scores on the test set.

    ################################# STUDENT SOLUTION ##########################
    #############################################################################