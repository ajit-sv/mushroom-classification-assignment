import numpy as np

def confusion_matrix(y_true, y_pred):
    tp = np.sum((y_true==1) & (y_pred==1))
    tn = np.sum((y_true==0) & (y_pred==0))
    fp = np.sum((y_true==0) & (y_pred==1))
    fn = np.sum((y_true==1) & (y_pred==0))
    return tp, tn, fp, fn

def accuracy(tp, tn, fp, fn):
    return (tp+tn)/(tp+tn+fp+fn)

def precision(tp, fp):
    return tp/(tp+fp+1e-10)

def recall(tp, fn):
    return tp/(tp+fn+1e-10)

def f1_score(p, r):
    return 2*p*r/(p+r+1e-10)

def mcc(tp, tn, fp, fn):
    num = (tp*tn)-(fp*fn)
    den = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    return num/(den+1e-10)

# AOC (AUC) computation for binary classification
def aoc_score(y_true, y_pred_proba, pos_label=1, debug=False):
    """
    Compute Area Over Curve (AUC) using the trapezoidal rule.
    y_true: array of true labels (0/1 or int codes)
    y_pred_proba: array of predicted probabilities for positive class
    pos_label: integer code for positive class
    debug: if True, print debug info
    """
    # Sort by predicted probability descending
    desc_order = np.argsort(-y_pred_proba)
    y_true_sorted = np.array(y_true)[desc_order]
    y_pred_sorted = np.array(y_pred_proba)[desc_order]
    # True/False positive rates
    P = np.sum(y_true_sorted == pos_label)
    N = np.sum(y_true_sorted != pos_label)
    tpr = []
    fpr = []
    tp = fp = 0
    for i in range(len(y_true_sorted)):
        if y_true_sorted[i] == pos_label:
            tp += 1
        else:
            fp += 1
        tpr.append(tp / (P+1e-10))
        fpr.append(fp / (N+1e-10))
    # Add (0,0) at start
    tpr = [0.0] + tpr
    fpr = [0.0] + fpr
    # Trapezoidal rule
    auc = 0.0
    for i in range(1, len(tpr)):
        auc += (fpr[i] - fpr[i-1]) * (tpr[i] + tpr[i-1]) / 2
    if debug:
        print(f"AOC Debug: P={P}, N={N}")
        print(f"y_true_sorted[:10]={y_true_sorted[:10]}")
        print(f"y_pred_sorted[:10]={y_pred_sorted[:10]}")
        print(f"TPR[:10]={tpr[:10]}")
        print(f"FPR[:10]={fpr[:10]}")
        print(f"AOC (AUC)={auc}")
    return auc