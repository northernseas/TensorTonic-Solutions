import numpy as np

def auc_roc(y_true, y_scores):
    """
    Returns: tuple of (fpr_list, tpr_list, auc_value)
    """
    y_true = np.array(y_true)
    y_scores = np.array(y_scores, dtype=np.float64)

    idx = np.argsort(y_scores)[::-1]
    y_true = y_true[idx]
    y_scores = y_scores[idx]

    thresholds = np.unique(y_scores)[::-1]
    
    tpr = [0]
    fpr = [0]
    
    for threshold in thresholds:
        y_pred = y_scores >= threshold

        tp = np.sum((y_pred == 1) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        
        tpr.append(tp / (tp + fn))
        fpr.append(fp / (fp + tn))
        
    auc = 0
    for i in range(1, len(tpr)):
        auc += (fpr[i] - fpr[i - 1]) * (tpr[i] + tpr[i - 1]) / 2
    
    return np.round(fpr, 4), np.round(tpr, 4), round(auc, 4)