import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from utils.metrics import evaluate_model

def run_lightgbm_baseline(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> None:
    """Run and evaluate LightGBM baseline."""
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError("lightgbm not installed. Run: pip install lightgbm")

    clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, random_state=1, verbose=-1)
    clf.fit(x_train, y_train)
    train_preds = clf.predict(x_train)
    train_metrics = {
        'acc': accuracy_score(y_train, train_preds),
        'precision': precision_score(y_train, train_preds),
        'recall': recall_score(y_train, train_preds),
        'fscore': f1_score(y_train, train_preds),
        'mcc': matthews_corrcoef(y_train, train_preds),
    }
    y_pred = clf.predict(x_test)
    val_precision, val_recall, val_fscore, val_mcc, val_acc = evaluate_model(y_test, y_pred)
    print('LightGBM baseline:')
    print(f'T_acc {train_metrics["acc"]:.3f}\tT_pre {train_metrics["precision"]:.3f}\tT_rec {train_metrics["recall"]:.3f}\tT_fscore {train_metrics["fscore"]:.3f}\tT_mcc {train_metrics["mcc"]:.3f}')
    print(f'V_acc {val_acc:.3f}\tV_pre {val_precision:.3f}\tV_rec {val_recall:.3f}\tV_fscore {val_fscore:.3f}\tV_mcc {val_mcc:.3f}')
