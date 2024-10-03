# NLP_Language-Modeling-and-Text-Prediction


# Natural Language Processing - Language Modeling and Text Prediction

## Overview

This project focuses on building and evaluating a language model using a neural network to perform text prediction based on a corpus from Charles Dickens' "David Copperfield." The goal is to understand and apply the principles of statistical language models and train a custom model using neural network-based approaches. The assignment also explores generating coherent text sequences using the trained model.

## Tasks

1. **Data Preprocessing**: 
   - Tokenize and prepare the input text data from "David Copperfield" for training.
   - Create vocabulary mappings and split the dataset into training and validation sets.

2. **Model Architecture**:
   - Develop a Feedforward Neural Network (FFNN) architecture suitable for language modeling.
   - Implement a simple RNN-based language model for sequential text generation.

3. **Training**:
   - Train the model on the provided dataset (`dickens_train.txt`) and evaluate its performance using validation data (`dickens_test.txt` and `dickens_test_large.txt`).

4. **Evaluation**:
   - Measure the performance of the model using perplexity and loss functions.
   - Compare the predictions of the trained model with human-readable text samples.

## Files

- `assignment3.py`: Main script containing the preprocessing, model training, and text generation logic.
- `evaluation.py`: Script for evaluating the model’s performance on test datasets.
- `language_model.py`: Implementation of the language model architecture (RNN).
- `utils.py`: Utility functions used throughout the project, including text processing and dataset handling.
- `dickens_train.txt`: Training dataset from Charles Dickens' *David Copperfield*.
- `dickens_test.txt`: Small test dataset used for evaluating the model's predictions.
- `dickens_test_large.txt`: Larger test dataset for extensive evaluation of the model's performance.

## How to Run

1. Install dependencies (if any):
   ```bash
   pip install -r requirements.txt
   ```

2. To train the model, run:
   ```bash
   python assignment3.py --train dickens_train.txt --test dickens_test.txt
   ```

3. To evaluate the model, use:
   ```bash
   python evaluation.py --model <model_path> --test dickens_test_large.txt
   ```

## Results

The model was evaluated on the validation dataset using perplexity and loss. While the neural network successfully learned to generate text sequences, improvements in hyperparameters, dataset size, and architecture can further enhance the model's text prediction capabilities.
