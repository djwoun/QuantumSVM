# -*- coding: utf-8 -*-
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import genfromtxt
from sklearn import svm, datasets
import random


#X, y, centers = datasets.make_blobs(n_samples=100, centers=2, n_features=2,cluster_std=0.68,
 #                          center_box=(0,10),return_centers=True)
#y = y.reshape((y.shape[0], 1))

#plt.scatter(X[:, 0],X[:, 1] ,c=y, s=30, cmap=plt.cm.Paired)
#print(centers)
standardDeviation = [0.001]
#features = 2 ** np.arange(22)
#features = np.delete(features,0)
features = [2**4]
#print(features)
#print(features)



for feat in features:
    for std in standardDeviation:
        for x in range(1):
            X, y = datasets.make_blobs(n_samples=256, centers=2, n_features=feat,cluster_std=std,
                                       random_state=x,center_box=(0,4))
            
            
            print(X.astype(int))
            print(y)
            #plt.scatter(X[:, 0],X[:, 1] ,c=y, s=30, cmap=plt.cm.Paired)
            #plt.legend(title='Features', bbox_to_anchor=(1, 1), loc='upper left')
           # plt.show()
            
        #print(y)
            #plt.savefig(str(x)+'.png', bbox_inches='tight', dpi=250)
            #plt.clf()
            #print(X)
            ran = random.sample(range(256), int(77))
            #np.savetxt('SynCsvFile0.001/X'+str(x)+' '+str(feat)+'.csv', X, delimiter=',') 
            #np.savetxt('SynCsvFile0.001/Y'+str(x)+' '+str(feat)+'.csv', y, delimiter=',') 
            #np.savetxt('SynCsvFile0.001/random'+str(x)+' '+str(feat)+'.csv', ran, delimiter=',') 
        
        #print(y)
