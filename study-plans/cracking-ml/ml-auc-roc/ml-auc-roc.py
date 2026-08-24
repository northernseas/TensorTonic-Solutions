import numpy as np

def auc_roc(y_true, y_scores):
    """
    Returns: tuple of (fpr_list, tpr_list, auc_value)
    """
    y_true = np.array(y_true, dtype=bool)
    y_scores = np.array(y_scores, dtype=np.float64)

    thresholds = np.unique(y_scores)[::-1]
    
    tpr = [0]
    fpr = [0]

    nb_positives = np.sum(y_true)
    nb_negatives = np.sum(~y_true)
    
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(bool)

        tp = np.sum(y_pred & y_true)
        fp = np.sum(y_pred & ~y_true)

        tpr.append(tp / nb_positives)
        fpr.append(fp / nb_negatives)
        
    auc = 0
    for i in range(1, len(tpr)):
        auc += (fpr[i] - fpr[i - 1]) * (tpr[i] + tpr[i - 1]) / 2
    
    return np.round(fpr, 4), np.round(tpr, 4), round(auc, 4)