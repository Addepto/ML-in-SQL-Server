from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, recall_score, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix, auc, precision_recall_curve, roc_curve
import numpy as np

#Make prediction
def predict(model, X_test, y_test, model_name="xgb"): 
    if model_name == "rfc":
        y_score = model.predict_proba(X_test)[:,1]
    elif model_name == "xgb":
        y_score = model.predict_proba(X_test, ntree_limit=model.best_ntree_limit)[:,1]

    average_precision = average_precision_score(y_test.values, y_score)
    precision, recall, thr = precision_recall_curve(y_test.values, y_score)

    f = 2*precision*recall / (precision+recall)
    f_max_arg = np.where(f == max(f))
    predictions = np.where(y_score >= thr[f_max_arg], 1, 0)
    roc_auc = roc_auc_score(y_test, y_score, average=None)
    
    print()
    print("AUC ROC: \t",   round(roc_auc,3))
    print("GINI:    \t",   round(2*roc_auc-1,3))
    print("F1_max: \t",    round(max(f), 3))
    print()
    print("For F1_max:")
    print("Threshold: \t", round(thr[f_max_arg][0], 3))
    print("Recall: \t",    round(recall[f_max_arg][0], 3))
    print("Precision: \t", round(precision[f_max_arg][0], 3))
    print("Accuracy: \t",  round(accuracy_score(y_test, predictions), 3)) 
    
    print()
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))
    
    print()
    print("Classification report:")
    print(classification_report(y_test, predictions))
    
    return y_score, roc_auc, 2*roc_auc-1, max(f), recall[f_max_arg][0], precision[f_max_arg][0], accuracy_score(y_test, predictions), thr[f_max_arg][0]