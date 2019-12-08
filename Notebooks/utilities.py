import folium
import collections
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score,\
                            roc_curve,precision_recall_curve, auc, roc_curve

def plot_model_scores(clfs_with_data_list):

    perf_graph = collections.namedtuple('performance_graphs', ['X_coordinates', 'y_coordinates', 'auc', 'label'])
    container_rocs = []
    container_prs = []
    
    for clf_with_data in clfs_with_data_list:
        
        clf = clf_with_data.clf
        
        y_score = clf.predict_proba(clf_with_data.X)[:, 1]
        roc_data_points = roc_curve(clf_with_data.y, y_score)
        roc_auc = roc_auc_score(clf_with_data.y, y_score) 
        plotting_data=perf_graph(roc_data_points[0], roc_data_points[1], roc_auc, clf_with_data.name)
        container_rocs.append(plotting_data)
        
        pr_data_points = precision_recall_curve(clf_with_data.y, y_score)
        pr_auc = auc(pr_data_points[1], pr_data_points[0])
        plotting_data_pr = perf_graph(pr_data_points[0], pr_data_points[1], pr_auc, clf_with_data.name)
        container_prs.append(plotting_data_pr)
        
    fig, ax = plt.subplots(1,2, figsize=(10,5))
     

    ax[0].set_aspect('equal')
    ax[0].set_title('ROC Kurve')
    ax[0].set_ylabel(u'TPR')
    ax[0].set_xlabel(u'FPR')
    ax[0].set_ylim([0.0,1.0])
    ax[0].set_xlim([0.0,1.0])

    ax[1].set_aspect('equal')
    ax[1].set_title('Precision-recall Kurve')
    ax[1].set_ylabel(u'precision')
    ax[1].set_xlabel(u'recall')
    ax[1].set_ylim([0.0,1.0])
    ax[1].set_xlim([0.0,1.0])

        
    for element in container_rocs :
        ax[0].plot(element.X_coordinates, element.y_coordinates, label=element.label+' auc: {}'.format(element.auc))
  
    for element in container_prs:
        ax[1].plot(element.X_coordinates, element.y_coordinates, label=element.label+' auc: {}'.format(element.auc))
           
    ax[0].plot([0,1], [0,1])
    ax[1].plot([0,1], [0,1])
    ax[0].legend(loc=0)
    ax[1].legend(loc=0)
    #fig.tight_layout()
    plt.show()
    #fig_show()

def setup_and_plot_model_scores(clf, X_test, y_test, X_train, y_train):

    perf_data= collections.namedtuple('performance_data', ['clf','X', 'y', 'name'])
    train_data = perf_data(clf, X_train, y_train, 'train')
    test_data = perf_data(clf, X_test, y_test, 'test')
    print('\n__________________ \t\t\t\t', str(clf.__class__))
    plot_model_scores([train_data, test_data])
