# -*- coding: utf-8 -*-
import csv
import numpy as np
import pandas as pd
import random
from sklearn import svm
a = [0,1,0,0,1,0,0,1,0]
a2 = [1,0,0,1,0,0,1,0,0]
a3 = [0,0,1,0,0,1,0,0,1]
a4 = [1,1,0,1,1,0,1,1,0]
a5 = [0,1,1,0,1,1,0,1,1]
a6 = [1,0,1,1,0,1,1,0,1]
a7= [1,1,1,0,0,0,0,0,0]
a8 = [0,0,0,1,1,1,0,0,0]
a9 = [0,0,0,0,0,0,1,1,1]
a10 = [1,1,1,1,1,1,0,0,0]
a11 = [0,0,0,1,1,1,1,1,1]
a12 = [1,1,1,0,0,0,1,1,1]
b = [1,1,1,1,1,1,-1,-1,-1,-1,-1,-1]
c = np.ones(50 )
d = -1*np.ones(50 )
w =  np.concatenate((c, d), axis=0)
cStack = np.vstack([a,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12])
aStack = np.vstack([a,a2,a3,a4,a5,a6])
bStack = np.vstack([a7,a8,a9,a10,a11,a12])
AStack = pd.DataFrame(aStack)
BStack = pd.DataFrame(bStack)
for x in range (30):
    
    X = AStack.sample(50,replace=True)
    Y = BStack.sample(50,replace=True)
    
    ran = random.sample(range(100), int(30))
    
    
    np.savetxt('BNSCSVFiles/BN'+str(x)+'.csv', X, delimiter=',') 
    np.savetxt('BNSCSVFiles/NS'+str(x)+'.csv', Y, delimiter=',') 
    np.savetxt('BNSCSVFiles/random'+str(x)+'.csv', ran, delimiter=',')
    print (ran)
    
    clf = svm.SVC(kernel="poly", C=1,degree=3)
    
    #start = time.process_time() 
    clf.fit(cStack, b)
    aStack = np.concatenate((X,Y), axis=0)
    #clf = svm.SVC(kernel="rbf", C=1,degree =3)
    #end = time.process_time() 
    print(clf.score(aStack,w))
     
    #print(clf.predict(aStack))