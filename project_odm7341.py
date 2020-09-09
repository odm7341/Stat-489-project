# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 12:02:13 2020

@author: omccl
"""


import matplotlib.pyplot as plt
import numpy as np 
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score


def makeGph(xDat, yDat):
        #print(xDat[:,0])
    plt.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots(1, 7, sharey=True, sharex=True)
    #fig.suptitle('AP RSSI vs Room Number')
    for subplot in ax:
        subplot.set_xlim(.9,4.1)
        subplot.set_ylim(-100,0)
        subplot.set_xticks([1, 2, 3, 4])
    ax[0].plot(yDat, xDat[:, 0], 'C0.')
    ax[0].set_title("AP 1")
    ax[1].plot(yDat, xDat[:, 1], 'C1.')
    ax[1].set_title("AP 2")
    ax[2].plot(yDat, xDat[:, 2], 'C2.')
    ax[2].set_title("AP 3")
    ax[3].plot(yDat, xDat[:, 3], 'C3.')
    ax[3].set_title("AP 4")
    ax[4].plot(yDat, xDat[:, 4], 'C4.')
    ax[4].set_title("AP 5")
    ax[5].plot(yDat, xDat[:, 5], 'C5.')
    ax[5].set_title("AP 6")
    ax[6].plot(yDat, xDat[:, 6], 'C6.')
    ax[6].set_title("AP 7")
    fig.set_size_inches(8, 2)
    fig.savefig('gph.png', dpi=300)
    '''
    plt.plot(xDat, yDat, '.')
    plt.rcParams.update({'font.size': 7})
    plt.yticks([1, 2, 3, 4])
    plt.axes().set_aspect(8)
    plt.legend(["AP 1", "AP 2", "AP 3", "AP 4", "AP 5", "AP 6", "AP 7"])
    plt.title("Room Number vs AP RSSI")
    plt.savefig('gph.png', dpi=300)
    '''

def KNNReg(X_train, X_test, y_train, y_test):
    neighbor = [1, 3, 5, 7, 9]
    cv_score = []
    for k in range(0,5):
        knn_model=KNeighborsClassifier(n_neighbors=neighbor[k],p=2,metric='minkowski')
        score=cross_val_score(knn_model,X_train,y_train,cv=5,scoring='accuracy')
        cv_score.append(np.mean(score))
    best = np.argmax(cv_score)
    print("\nKNN Results")
    print("Best k =", neighbor[best])
    print("Score =", cv_score[best]*100)
    plt.plot(neighbor, cv_score, "C2.")
    #plt.xscale("log")
    plt.xlabel("k value")
    plt.ylabel("Model Score")
    plt.title("Finding Best k")
    plt.savefig('findK.png', dpi=250)

def svmReg(X_train, X_test, y_train, y_test):
    
    Cs = []
    scores_gau = []        

    for i in range(29):
        #I chose this because its a nice range from 1e-5 to 1000
        Cs.append(1e-5 * (1.75 ** i) )

    for c in Cs:
        model = svm.SVC( C=c, kernel = 'rbf' )# <-- change to linear to see the linear kernel function scores
        model.fit(X_train, y_train)
        scores_gau.append(model.score(X_test, y_test))
        
    bestI_gau = np.argmax(scores_gau)
    print("\nSVM Results")
    print("Best validating C =", Cs[bestI_gau], ", Score =", scores_gau[bestI_gau]*100)
    plt.plot(Cs, scores_gau, "C2.")
    plt.xscale("log")
    plt.xlabel("Parameter")
    plt.ylabel("Model Score")
    plt.title("Finding Regularization Parameter")
    plt.savefig('regparam.png', dpi=300)
    

if __name__ == "__main__":
    
    Alldata = np.loadtxt('wifi_localization.txt')
    
    yDat = Alldata[:,7]
    xDat = Alldata[:,0:7]
    
    #makeGph(xDat, yDat)
    
    X_train, X_test, y_train, y_test = train_test_split(xDat, yDat, train_size=0.67)
    
    #svmReg(X_train, X_test, y_train, y_test)
    KNNReg(X_train, X_test, y_train, y_test)
    
    
    myData = np.loadtxt('mydata.txt')
    
    yDat2 = myData[:,7]
    xDat2 = myData[:,0:7]
    
    #makeGph(xDat2, yDat2)
    
    X_train, X_test, y_train, y_test = train_test_split(xDat2, yDat2, train_size=0.67)
    
    #svmReg(X_train, X_test, y_train, y_test)
    #KNNReg(X_train, X_test, y_train, y_test)
    
    
    
