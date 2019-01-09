# Sparse-LInear-Models (SLIM)
SLIM model for top_n recommendation

This is a C++ implementation of the top_n recommendation algorithm SLIM from the paper

Ning, Xia, and George Karypis. "Slim: Sparse linear methods for top-n recommender systems." 2011 11th IEEE International Conference on Data Mining. IEEE, 2011.

# Requirements
* gcc
* OpenMP

# Usage
1. Type make to compile the code and generate the excutable
2. Run <code>./slim data_file l1 l2 top_n num_threads eps</code>

# Args
- <code>data_file</code>: path to the dataset
- <code>l1</code>: hyperparameter for l1 regularization which controls the sparsity of the model
- <code>l2</code>: hyperparameter for l2 regularization
- <code>top_n</code>: length of recommended lists
- <code>num_threads</code>: number of threads available to accelerate the training process
- <code>eps</code>: number to control the convergence of the model

# Data format
The dataset shoud have three columns and be formated as 

> user_id, item_id, timestamp

A sample dataset amazon_game_5.csv is uploaded.
