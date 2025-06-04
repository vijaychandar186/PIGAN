import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import time
from data.loading import load_dataset
from data.processing import reshape_to_linear_features
from models.rnn import RNNModel
from models.attention import AttentionModel, DualAttentionRNNModel
from models.transformer import TransformerModel
from models.mutagan import MutaGANModel
from utils.metrics import evaluate_model
from experiments.training import train_rnn_model
from utils.visualization import format_time
from baselines.svm import run_svm_baseline
from baselines.random_forest import run_random_forest_baseline
from baselines.logistic_regression import run_logistic_regression_baseline
from baselines.knn import run_knn_baseline
from baselines.naive_bayes import run_bayes_baseline

def run_experiment(model_type: str, subtype: str, data_path: str, dataset_path: str) -> None:
    """Run the mutation prediction experiment for a given model and subtype."""
    parameters = {
        'dataset_path': dataset_path,
        'data_path': data_path,
        'model_type': model_type,
        'hidden_size': 512,
        'dropout_p': 0.0001,
        'learning_rate': 0.001,
        'batch_size': 256,
        'num_epochs': 100
    }

    torch.manual_seed(1)
    np.random.seed(1)

    # Load data
    train_vectors, train_labels = load_dataset(f"{parameters['dataset_path']}_train.csv", parameters['data_path'], concat=False)
    test_vectors, test_labels = load_dataset(f"{parameters['dataset_path']}_test.csv", parameters['data_path'], concat=False)

    x_train = torch.tensor(train_vectors, dtype=torch.float32)
    y_train = torch.tensor(train_labels, dtype=torch.int64)
    x_test = torch.tensor(test_vectors, dtype=torch.float32)
    y_test = torch.tensor(test_labels, dtype=torch.int64)

    # Print class imbalance
    _, train_counts = np.unique(y_train, return_counts=True)
    train_imbalance = max(train_counts) / y_train.shape[0]
    _, test_counts = np.unique(y_test, return_counts=True)
    test_imbalance = max(test_counts) / y_test.shape[0]
    print('Class imbalances:')
    print(f' Training {train_imbalance:.3f}')
    print(f' Testing  {test_imbalance:.3f}')

    # Run baseline models
    if parameters['model_type'] in ['svm', 'random forest', 'logistic regression', 'knn', 'naive bayes']:
        window_size = 1
        x_train_linear = reshape_to_linear_features(train_vectors, window_size=window_size)
        x_test_linear = reshape_to_linear_features(test_vectors, window_size=window_size)
        if parameters['model_type'] == 'svm':
            run_svm_baseline(x_train_linear, train_labels, x_test_linear, test_labels)
        elif parameters['model_type'] == 'random forest':
            run_random_forest_baseline(x_train_linear, train_labels, x_test_linear, test_labels)
        elif parameters['model_type'] == 'logistic regression':
            run_logistic_regression_baseline(x_train_linear, train_labels, x_test_linear, test_labels)
        elif parameters['model_type'] == 'knn':
            run_knn_baseline(x_train_linear, train_labels, x_test_linear, test_labels)
        elif parameters['model_type'] == 'naive bayes':
            run_bayes_baseline(x_train_linear, train_labels, x_test_linear, test_labels)
    else:
        # Initialize neural network model
        input_dim = x_train.shape[2]
        seq_length = x_train.shape[0]
        output_dim = 2
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        model_map = {
            'lstm': lambda: RNNModel(input_dim, output_dim, parameters['hidden_size'], parameters['dropout_p'], 'LSTM'),
            'gru': lambda: RNNModel(input_dim, output_dim, parameters['hidden_size'], parameters['dropout_p'], 'GRU'),
            'rnn': lambda: RNNModel(input_dim, output_dim, parameters['hidden_size'], parameters['dropout_p'], 'RNN'),
            'attention': lambda: AttentionModel(seq_length, input_dim, output_dim, parameters['hidden_size'], parameters['dropout_p']),
            'da-rnn': lambda: DualAttentionRNNModel(seq_length, input_dim, output_dim, parameters['hidden_size'], parameters['dropout_p']),
            'transformer': lambda: TransformerModel(100, 2, parameters['dropout_p']),
            'mutagan': lambda: MutaGANModel(input_dim, output_dim, parameters['hidden_size'], num_layers=2, dropout_p=parameters['dropout_p'], device=device)
        }
        
        model = model_map[parameters['model_type']]()
        model = model.to(device)
        x_train = x_train.to(device)
        y_train = y_train.to(device)
        x_test = x_test.to(device)
        y_test = y_test.to(device)
        
        train_rnn_model(
            model, verify=False, epochs=parameters['num_epochs'], learning_rate=parameters['learning_rate'],
            batch_size=parameters['batch_size'], x_train=x_train, y_train=y_train,
            x_test=x_test, y_test=y_test, show_attention=True, cell_type=parameters['model_type']
        )