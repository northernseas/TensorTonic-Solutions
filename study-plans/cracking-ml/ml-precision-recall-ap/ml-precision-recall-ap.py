import numpy as np

def precision_recall_ap(y_true, y_scores):
    """
    Returns: tuple of (recall_list, precision_list, ap_value)
    """
    # Recall = TPR = TP / (TP + FN)
    # Precision = TP / (TP + FP)

    y_true = np.array(y_true, dtype=bool)
    y_scores = np.array(y_scores, dtype=np.float64)

    thresholds = np.unique(y_scores)[::-1]

    recall_list = [0]
    precision_list = [1]

    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(bool)

        tp = np.sum(y_pred & y_true)
        fp = np.sum(y_pred & ~y_true)
        tn = np.sum(~y_pred & ~y_true)
        fn = np.sum(~y_pred & y_true)
        
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)

        recall_list.append(recall)
        precision_list.append(precision)

    ap = 0
    for k in range(1, len(recall_list)):
        ap += (recall_list[k] - recall_list[k - 1]) * precision_list[k]

    return (
        np.round(recall_list, 4),
        np.round(precision_list, 4),
        round(ap, 4)
    )