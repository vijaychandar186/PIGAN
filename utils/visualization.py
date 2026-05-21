import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np


def format_time(seconds):
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:>2}m {seconds:>2}s"


def print_value_counts(name, data):
    print(name)
    unique, counts = np.unique(data, return_counts=True)
    print(dict(zip(unique, counts)))


def plot_training_metrics(loss, val_loss, acc, val_acc, fscore, val_fscore):
    plt.style.use('ggplot')
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    for ax, train, val, title in [
        (ax1, loss, val_loss, 'Loss'),
        (ax2, acc, val_acc, 'Accuracy'),
        (ax3, fscore, val_fscore, 'F-Score'),
    ]:
        ax.plot(train, 'b', label='Training')
        ax.plot(val, 'r', label='Validation')
        ax.set_title(title)
        ax.legend()
    plt.show()


def plot_attention_weights(weights):
    cax = plt.matshow(weights.numpy(), cmap='bone')
    plt.colorbar(cax)
    plt.grid(False)
    plt.xlabel('Years')
    plt.ylabel('Examples')
    plt.show()


def detach_hidden(hidden):
    if isinstance(hidden, torch.Tensor):
        return hidden.detach()
    return tuple(detach_hidden(v) for v in hidden)


def get_predictions(scores):
    _, predictions = F.softmax(scores, dim=1).topk(1)
    return predictions


def calculate_probabilities(scores):
    pred_prob, _ = F.softmax(scores, dim=1).topk(1)
    return pred_prob
