import argparse
import torch
import torch.nn as nn
import unidecode
import string
import time

from utils import char_tensor, random_training_set, time_since, CHUNK_LEN
from language_model import plot_loss, diff_temp, custom_train, train_step, generate, tuner
from model.model import LSTM


def main():
    parser = argparse.ArgumentParser(
        description='Train LSTM'
    )

    parser.add_argument(
        '--default_train', dest='default_train',
        help='Train LSTM with default hyperparameter',
        action='store_true'
    )

    parser.add_argument(
        '--custom_train', dest='custom_train',
        help='Train LSTM while tuning hyperparameter',
        action='store_true'
    )

    parser.add_argument(
        '--plot_loss', dest='plot_loss',
        help='Plot losses chart with different learning rates',
        action='store_true'
    )

    parser.add_argument(
        '--diff_temp', dest='diff_temp',
        help='Generate strings by using different temperature',
        action='store_true'
    )

    args = parser.parse_args()

    all_characters = string.printable
    n_characters = len(all_characters)

    print("All characters: ", all_characters)
    print("n_characters: ", n_characters)

    if args.default_train:
        n_epochs = 3000
        print_every = 100
        plot_every = 10
        hidden_size = 128
        n_layers = 2

        lr = 0.005
        decoder = LSTM(n_characters, hidden_size,n_characters, n_layers)
        decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)

        start = time.time()
        all_losses = []
        loss_avg = 0

        hidden, cell = decoder.init_hidden()

        for epoch in range(1, n_epochs+1):
            print(f"Starting Epoch {epoch}/{n_epochs}")

            inp, target = random_training_set()

            loss = train_step(decoder, decoder_optimizer, inp, target)
            print(f"Epoch {epoch}, Loss: {loss}")

            #loss_avg += loss
            hidden, cell = decoder.init_hidden()

            if epoch % print_every == 0:
                elapsed_time = time_since(start)
                print(f"[Time: {elapsed_time}] Epoch {epoch}/{n_epochs} - Loss: {loss:.4f}")
                print(f"Sample Output: {generate(decoder, 'A', 100)}\n")
                # print('[{} ({} {}%) {:.4f}]'.format(time_since(start), epoch, epoch/n_epochs * 100, 0.5))
                # print(generate(decoder, 'A', 100), '\n')

            if epoch % plot_every == 0:
                all_losses.append(loss_avg / plot_every)
                loss_avg = 0

    if args.custom_train:
        # YOUR CODE HERE
        #     TODO:
        #         1) Fill in `hyperparam_list` with dictionary of hyperparameters
        #         that you want to try.
        ####################### STUDENT SOLUTION ###############################
        hyperparam_list = [{'hidden_size': 128, 'n_layers': 2, 'lr': 0.005},
    {'hidden_size': 256, 'n_layers': 2, 'lr': 0.003},
    {'hidden_size': 128, 'n_layers': 3, 'lr': 0.001}]
        ########################################################################
        bpc = custom_train(hyperparam_list)

        for params, bpc_score in bpc.items():
            print(f"BPC for params {params}: {bpc_score:.4f}")

        # for keys, values in bpc.items():
        #     print("BPC {}: {}".format(keys, values))

    if args.plot_loss:
        # YOUR CODE HERE
        #     TODO:
        #         1) Fill in `lr_list` with learning rates that you want to try.
        ######################### STUDENT SOLUTION #############################
        lr_list = [0.005, 0.003, 0.001, 0.0005]
        print("Plotting training losses for different learning rates...")

        ########################################################################
        plot_loss(lr_list)

    if args.diff_temp:
        # YOUR CODE HERE
        #     TODO:
        #         1) Fill in `temp_list` with temperatures that you want to try.
        ########################### STUDENT SOLUTION ###########################
        temp_list = [0.2, 0.5, 0.8, 1.0]
        print("Generating text with different temperatures:")
        # Ensure the model is trained and assigned to decoder before this step
        decoder = tuner()  # Add this line to train the model
        for temp in temp_list:
            generated_text = generate(decoder, temperature=temp)
            print(f"Temperature {temp}: {generated_text}\n")

        ########################################################################
        diff_temp(temp_list)


if __name__ == "__main__":
    main()
