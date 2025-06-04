import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import numpy as np

def format_time(seconds: float) -> str:
    """Format time in seconds to a string in the format 'Xm Ys'."""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:>2}m {seconds:>2}s"

def print_value_counts(name: str, data: np.ndarray) -> None:
    """Print the frequency of unique values in a dataset."""
    print(name)
    unique, counts = np.unique(data, return_counts=True)
    print(dict(zip(unique, counts)))

def plot_training_metrics(loss: list, val_loss: list, acc: list, val_acc: list, fscore: list, val_fscore: list) -> None:
    """Plot training and validation metrics."""
    plt.style.use('ggplot')
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    ax1.plot(loss, 'b', label='Training')
    ax1.plot(val_loss, 'r', label='Validation')
    ax1.set_title('Loss')
    ax1.legend()
    
    ax2.plot(acc, 'b', label='Training')
    ax2.plot(val_acc, 'r', label='Validation')
    ax2.set_title('Accuracy')
    ax2.legend()
    
    ax3.plot(fscore, 'b', label='Training')
    ax3.plot(val_fscore, 'r', label='Validation')
    ax3.set_title('F-Score')
    ax3.legend()
    
    plt.show()

def plot_attention_weights(weights: torch.Tensor) -> None:
    """Plot attention weights as a heatmap."""
    cax = plt.matshow(weights.numpy(), cmap='bone')
    plt.colorbar(cax)
    plt.grid(False)
    plt.xlabel('Years')
    plt.ylabel('Examples')
    plt.show()

def detach_hidden(hidden: torch.Tensor) -> torch.Tensor:
    """Detach hidden state tensors to prevent gradient tracking."""
    if isinstance(hidden, torch.Tensor):
        return hidden.detach()
    return tuple(detach_hidden(v) for v in hidden)

def get_predictions(scores: torch.Tensor) -> torch.Tensor:
    """Convert model scores to predictions."""
    prob = F.softmax(scores, dim=1)
    _, predictions = prob.topk(1)
    return predictions

def calculate_probabilities(scores: torch.Tensor) -> torch.Tensor:
    """Calculate prediction probabilities."""
    prob = F.softmax(scores, dim=1)
    pred_prob, _ = prob.topk(1)
    return pred_prob

def verify_model_gradient(model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int) -> None:
    """Verify model gradients and loss for sanity checks."""
    print('Sanity checks:')
    criterion = nn.CrossEntropyLoss()
    scores, _ = model(x, model.init_hidden(y.shape[0]))
    expected_loss = -math.log(1 / model.output_dim)
    print(f' Loss @ init {criterion(scores, y).item():.3f}, expected ~{expected_loss:.3f}')
    
    mini_batch_x = x[:, :batch_size, :].requires_grad_()
    criterion = nn.MSELoss()
    scores, _ = model(mini_batch_x, model.init_hidden(batch_size))
    non_zero_idx = 1
    perfect_scores = [[0, 0] for _ in range(batch_size)]
    not_perfect_scores = [[1, 1] if i == non_zero_idx else [0, 0] for i in range(batch_size)]
    scores.data = torch.FloatTensor(not_perfect_scores)
    y_perfect = torch.FloatTensor(perfect_scores)
    loss = criterion(scores, y_perfect)
    loss.backward()
    
    zero_tensor = torch.FloatTensor([0] * x.shape[2])
    for i in range(mini_batch_x.shape[0]):
        for j in range(mini_batch_x.shape[1]):
            if (mini_batch_x.grad[i, j] != zero_tensor).any():
                assert j == non_zero_idx, 'Input with loss set to zero has non-zero gradient.'
    print(' Backpropagated dependencies OK')