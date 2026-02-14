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