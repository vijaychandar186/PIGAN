import torch
import torch.nn as nn
import math
import time
from utils.metrics import compute_confusion_matrix, compute_accuracy, compute_precision, compute_recall, compute_f1_score, evaluate_model
from utils.visualization import detach_hidden, get_predictions, plot_training_metrics, format_time

def train_rnn_model(
    model: nn.Module,
    verify: bool,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    show_attention: bool,
    cell_type: str
) -> None:
    """Train an RNN-based model and evaluate performance."""
    print_interval = 10
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    num_examples = x_train.shape[1]
    num_batches = math.floor(num_examples / batch_size)
    
    if verify:
        verify_model_gradient(model, x_train, y_train, batch_size)
    
    metrics = {
        'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [],
        'train_precision': [], 'train_recall': [], 'train_fscore': [], 'val_fscore': [], 'train_mcc': []
    }
    best_metrics = {
        'val_loss': float('inf'), 'val_acc': 0.0, 'val_precision': 0.0,
        'val_recall': 0.0, 'val_fscore': 0.0, 'val_mcc': 0.0, 'epoch': 0
    }
    
    plot_batch_size = 10
    non_zero_idx = next(i for i, label in enumerate(y_test) if label)
    x_plot_batch = x_test[:, non_zero_idx:non_zero_idx + plot_batch_size, :]
    y_plot_batch = y_test[non_zero_idx:non_zero_idx + plot_batch_size]
    
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        running_metrics = {
            'loss': 0, 'acc': 0, 'precision': 0, 'precision_total': 0,
            'recall': 0, 'recall_total': 0, 'mcc_numerator': 0, 'mcc_denominator': 0
        }
        hidden = model.init_hidden(batch_size)
        
        for count in range(0, num_examples - batch_size + 1, batch_size):
            if hidden is not None:
                hidden = detach_hidden(hidden)
            x_batch = x_train[:, count:count + batch_size, :]
            y_batch = y_train[count:count + batch_size]
            scores, _ = model(x_batch, hidden)
            loss = criterion(scores, y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            predictions = get_predictions(scores)
            conf_matrix = compute_confusion_matrix(y_batch.cpu(), predictions.cpu())
            tp, fp, fn, tn = conf_matrix[0][0], conf_matrix[0][1], conf_matrix[1][0], conf_matrix[1][1]
            
            running_metrics['acc'] += tp + tn
            running_metrics['precision'] += tp
            running_metrics['precision_total'] += tp + fp
            running_metrics['recall'] += tp
            running_metrics['recall_total'] += tp + fn
            running_metrics['mcc_numerator'] += (tp * tn - fp * fn)
            running_metrics['mcc_denominator'] += math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) > 0 else 0
            running_metrics['loss'] += loss.item()
        
        elapsed_time = time.time() - start_time
        epoch_metrics = {
            'acc': running_metrics['acc'] / y_train.shape[0],
            'precision': running_metrics['precision'] / running_metrics['precision_total'] if running_metrics['precision_total'] > 0 else 0,
            'recall': running_metrics['recall'] / running_metrics['recall_total'] if running_metrics['recall_total'] > 0 else 0,
            'loss': running_metrics['loss'] / num_batches,
            'mcc': running_metrics['mcc_numerator'] / running_metrics['mcc_denominator'] if running_metrics['mcc_denominator'] > 0 else 0
        }
        epoch_metrics['fscore'] = 2 * epoch_metrics['precision'] * epoch_metrics['recall'] / (epoch_metrics['precision'] + epoch_metrics['recall']) if epoch_metrics['precision'] + epoch_metrics['recall'] > 0 else 0
        
        metrics['train_loss'].append(epoch_metrics['loss'])
        metrics['train_acc'].append(epoch_metrics['acc'])
        metrics['train_precision'].append(epoch_metrics['precision'])
        metrics['train_recall'].append(epoch_metrics['recall'])
        metrics['train_fscore'].append(epoch_metrics['fscore'])
        metrics['train_mcc'].append(epoch_metrics['mcc'])
        
        with torch.no_grad():
            model.eval()
            test_scores, _ = model(x_test, model.init_hidden(y_test.shape[0]))
            predictions = get_predictions(test_scores).view_as(y_test)
            val_precision, val_recall, val_fscore, val_mcc, val_acc = evaluate_model(y_test.cpu(), predictions.cpu())
            val_loss = criterion(test_scores, y_test).item()
            
            metrics['val_loss'].append(val_loss)
            metrics['val_acc'].append(val_acc)
            metrics['val_fscore'].append(val_fscore)
            
            if val_acc > best_metrics['val_acc']:
                best_metrics.update({
                    'val_loss': val_loss, 'val_acc': val_acc, 'val_precision': val_precision,
                    'val_recall': val_recall, 'val_fscore': val_fscore, 'val_mcc': val_mcc, 'epoch': epoch
                })
        
        if (epoch + 1) % print_interval == 0:
            print(f'Epoch {epoch} Time {format_time(elapsed_time)}')
            print(f'T_loss {epoch_metrics["loss"]:.3f}\tT_acc {epoch_metrics["acc"]:.3f}\tT_pre {epoch_metrics["precision"]:.3f}\tT_rec {epoch_metrics["recall"]:.3f}\tT_fscore {epoch_metrics["fscore"]:.3f}\tT_mcc {epoch_metrics["mcc"]:.3f}')
            print(f'V_loss {val_loss:.3f}\tV_acc {val_acc:.3f}\tV_pre {val_precision:.3f}\tV_rec {val_recall:.3f}\tV_fscore {val_fscore:.3f}\tV_mcc {val_mcc:.3f}')
    
    plot_training_metrics(
        metrics['train_loss'], metrics['val_loss'],
        metrics['train_acc'], metrics['val_acc'],
        metrics['train_fscore'], metrics['val_fscore']
    )
    print(f'Best results: Epoch {best_metrics["epoch"]} \n V_loss {best_metrics["val_loss"]:.3f}\tV_acc {best_metrics["val_acc"]:.3f}\tV_pre {best_metrics["val_precision"]:.3f}\tV_rec {best_metrics["val_recall"]:.3f}\tV_fscore {best_metrics["val_fscore"]:.3f}\tV_mcc {best_metrics["val_mcc"]:.3f}')