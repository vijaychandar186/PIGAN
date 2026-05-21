import torch
import numpy as np

from data.loading import load_dataset
from data.processing import reshape_to_linear_features
from models.rnn import RNNModel
from models.attention import AttentionModel, DualAttentionRNNModel
from models.transformer import TransformerModel
from models.mutagan import MutaGANModel
from experiments.training import train_rnn_model
from experiments.gan_training import run_pigan_experiment
from baselines.svm import run_svm_baseline
from baselines.random_forest import run_random_forest_baseline
from baselines.logistic_regression import run_logistic_regression_baseline
from baselines.knn import run_knn_baseline
from baselines.naive_bayes import run_bayes_baseline
from baselines.lightgbm_baseline import run_lightgbm_baseline
from baselines.gradient_boosting import run_gradient_boosting_baseline

BASELINES = {
    'svm': run_svm_baseline,
    'random forest': run_random_forest_baseline,
    'logistic regression': run_logistic_regression_baseline,
    'knn': run_knn_baseline,
    'naive bayes': run_bayes_baseline,
    'lightgbm': run_lightgbm_baseline,
    'gradient boosting': run_gradient_boosting_baseline,
}


def run_experiment(model_type, subtype, data_path, dataset_path):
    """Run a model on the given subtype."""
    parameters = {
        'hidden_size': 512,
        'dropout_p': 0.0001,
        'learning_rate': 0.001,
        'batch_size': 256,
        'num_epochs': 100,
    }

    torch.manual_seed(1)
    np.random.seed(1)

    train_vectors, train_labels = load_dataset(f"{dataset_path}_train.csv", data_path, concat=False)
    test_vectors, test_labels = load_dataset(f"{dataset_path}_test.csv", data_path, concat=False)

    x_train = torch.tensor(train_vectors, dtype=torch.float32)
    y_train = torch.tensor(train_labels, dtype=torch.int64)
    x_test = torch.tensor(test_vectors, dtype=torch.float32)
    y_test = torch.tensor(test_labels, dtype=torch.int64)

    _, train_counts = np.unique(y_train, return_counts=True)
    _, test_counts = np.unique(y_test, return_counts=True)
    print('Class imbalances:')
    print(f' Training {max(train_counts) / y_train.shape[0]:.3f}')
    print(f' Testing  {max(test_counts) / y_test.shape[0]:.3f}')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if model_type in BASELINES:
        x_train_linear = reshape_to_linear_features(train_vectors, window_size=1)
        x_test_linear = reshape_to_linear_features(test_vectors, window_size=1)
        BASELINES[model_type](x_train_linear, train_labels, x_test_linear, test_labels)
        return

    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)
    input_dim = x_train.shape[2]

    if model_type == 'pigan':
        run_pigan_experiment(
            x_train, y_train, x_test, y_test,
            input_dim=input_dim,
            hidden_size=256,
            num_layers=2,
            dropout_p=parameters['dropout_p'],
            pretrain_epochs=100,
            gan_epochs=30,
            g_lr=parameters['learning_rate'],
            d_lr=parameters['learning_rate'],
            batch_size=parameters['batch_size'],
        )
        return

    seq_length = x_train.shape[0]
    model_map = {
        'lstm':        lambda: RNNModel(input_dim, 2, parameters['hidden_size'], parameters['dropout_p'], 'LSTM'),
        'gru':         lambda: RNNModel(input_dim, 2, parameters['hidden_size'], parameters['dropout_p'], 'GRU'),
        'rnn':         lambda: RNNModel(input_dim, 2, parameters['hidden_size'], parameters['dropout_p'], 'RNN'),
        'attention':   lambda: AttentionModel(seq_length, input_dim, 2, parameters['hidden_size'], parameters['dropout_p']),
        'da-rnn':      lambda: DualAttentionRNNModel(seq_length, input_dim, 2, parameters['hidden_size'], parameters['dropout_p']),
        'transformer': lambda: TransformerModel(100, 2, parameters['dropout_p']),
        'mutagan':     lambda: MutaGANModel(input_dim, 2, parameters['hidden_size'], num_layers=2,
                                            dropout_p=parameters['dropout_p'], device=device),
    }
    model = model_map[model_type]().to(device)

    train_rnn_model(
        model, verify=False, epochs=parameters['num_epochs'],
        learning_rate=parameters['learning_rate'], batch_size=parameters['batch_size'],
        x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
        show_attention=True, cell_type=model_type,
    )
