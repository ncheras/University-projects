import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import scikitplot as skplt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from matplotlib import pyplot

def describe_evaluation(y_true, predictions):
    y_pred = predictions
    eval_acc = accuracy_score(y_true, y_pred)
    
    #Get the confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"Evaluation accuracy : {eval_acc}")
    
    skplt.metrics.plot_confusion_matrix(y_true, y_pred, normalize=True)
    skplt.metrics.plot_confusion_matrix(y_true, y_pred, normalize=False)

    #precision, recall, thresholds = precision_recall_curve(y_true, [pred[1] for pred in predictions], pos_label=True)
    lr_f1 = f1_score(y_true, y_pred, pos_label=True)

    #pyplot.figure()
    # summarize scores
    print('F1=%.3f' % (lr_f1))
    # plot the precision-recall curves
    # no_skill = y_true.count(True) / len(y_true)
    # pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    # pyplot.plot(recall, precision, marker='.', label='Logistic')
    # # axis labels
    # pyplot.xlabel('Recall')
    # pyplot.ylabel('Precision')
    # # show the legend
    # pyplot.legend()
    # # show the plot
    # pyplot.show()
    
    return eval_acc
      