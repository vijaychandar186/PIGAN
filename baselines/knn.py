from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from utils.metrics import evaluate_model
import numpy as np

def run_knn_baseline(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, method: str = None) -> None:
    """Run and evaluate k-Nearest Neighbors baseline."""
    clf = KNeighborsClassifier().fit(x_train, y_train)
    train_metrics = {
        'acc': accuracy_score(y_train, clf.predict(x_train)),
        'precision': precision_score(y_train, clf.predict(x_train)),
        'recall': recall_score(y_train, clf.predict(x_train)),
        'fscore': f1_score(y_train, clf.predict(x_train)),
        'mcc': matthews_corrcoef(y_train, clf.predict(x_train))
    }
    y_pred = clf.predict(x_test)
    val_precision, val_recall, val_fscore, val_mcc, val_acc = evaluate_model(y_test, y_pred)
    print('kNN baseline:')
    print(f'T_acc {train_metrics["acc"]:.3f}\tT_pre {train_metrics["precision"]:.3f}\tT_rec {train_metrics["recall"]:.3f}\tT_fscore {train_metrics["fscore"]:.3f}\tT_mcc {train_metrics["mcc"]:.3f}')
    print(f'V_acc {val_acc:.3f}\tV_pre {val_precision:.3f}\tV_rec {val_recall:.3f}\tV_fscore {val_fscore:.3f}\tV_mcc {val_mcc:.3f}')