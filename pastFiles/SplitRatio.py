# -*- coding: utf-8 -*-

from sklearn import svm, datasets
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.pyplot as plt
import numpy as np
import sys
import random
csvFile = sys.argv[1]
SplitRatio = sys.argv[2]
SplitRatio2 = 1-float(SplitRatio)
Noise = sys.argv[3]


iris = datasets.load_iris()

X =  iris.data[:, :2]
y =  iris.target


setosa = X [:50,:2]
versicolor = X [50:100, :2]
virginica = X [100:150, :2]

random.shuffle(setosa)
random.shuffle(versicolor)
#random.shuffle(virginica)

def svmFunc(X,y):
    clf = svm.SVC(kernel="linear", C=1)
    
    #%time clf.fit(X, y)
    clf.fit(X, y)
    
    print("weights: ", clf.coef_)
    print("bias: ",clf.intercept_)
    
    plt.scatter(X[:, 0],X[:, 1] ,c=y, s=30, cmap=plt.cm.Paired)

# plot the decision function
    ax = plt.gca()

    DecisionBoundaryDisplay.from_estimator(
        clf,
        X,
        plot_method="contour",
        colors="k",
        levels=[-1, 0, 1],
        alpha=0.5,
        linestyles=["--", "-", "--"],
        ax=ax,
    )

# plot support vectors

    ax.scatter(
        clf.support_vectors_[:, 0],
        clf.support_vectors_[:, 1],
        s=100,
        linewidth=1,
        facecolors="none",
        edgecolors="k",
    )

    plt.show()
    return clf.coef_, clf.intercept_

SplitRatioMDataA = int(float(SplitRatio) * setosa.size * 0.5)
SplitRatioMDataB = int(float(SplitRatio) * versicolor.size * 0.5)


#print(setosa.size)
#print(int(float(SplitRatio) * setosa.size))
a = setosa[:SplitRatioMDataA,:2]
b = versicolor[:SplitRatioMDataB,:2]
f = setosa[SplitRatioMDataA:int( setosa.size * 0.5 ),:2]
g = versicolor[SplitRatioMDataB:int( versicolor.size*0.5),:2]
q = np.concatenate((a, b), axis=0)


c = np.arange(0, SplitRatioMDataA*0.5,0.5, dtype=int)
d = np.ones(SplitRatioMDataB )
w =  np.concatenate((c, d), axis=0)
h = random.sample(range(int( setosa.size )), int(Noise))
ReverseSplitRatioMDataA = setosa.size - w.size

#temporary

j = np.arange(0, int(ReverseSplitRatioMDataA*0.5*0.5),0.5, dtype=int)
k = np.ones(int(ReverseSplitRatioMDataA/2))

'''
for x in h:
    if (i[x] == 1 ):
        i[x] = 0
    if(i[x] == 0):
        i[x] = 1
'''
print (h)

middle = w.size + ReverseSplitRatioMDataA/2


for x in h:
    if x<w.size:
        if (w[x] == 1 ):
            w[x] = 0
        elif(w[x] == 0):
            w[x] = 1
    if w.size<=x< middle:
        if (j[x-w.size] == 0 ):
            j[x-w.size] = 1
    if middle<=x< setosa.size:
        if(k[int(x-middle)] == 1):
            k[int(x-middle)] = 0
            
    
print(w)
print(j)
print(k)


weights, bias = svmFunc(q,w)

for x in h:
    if x<w.size:
        print (np.matmul(weights, q[x]) + bias)
    if w.size<=x< middle:
        print (np.matmul(weights, f[x-w.size]) + bias)
    if middle<=x< setosa.size:
        print (np.matmul(weights, g[int(x-middle)]) + bias)
    
    
        
testerror = 0
for x in range(int(f.size*0.5)):
    #print (np.matmul(weights, a[x]) + bias)
    if j[x] == 1:
        if (np.matmul(weights, f[x]) + bias <=0):
            testerror = testerror+1
    if j[x] == 0:
        if (np.matmul(weights, f[x]) + bias >=0):
            testerror = testerror+1

for x in range(int(g.size*0.5)):
    #print (np.matmul(weights, b[x]) + bias)
    if k[x] == 1:
        if (np.matmul(weights, g[x]) + bias <=0):
            testerror = testerror+1
    if k[x] == 0:
        if (np.matmul(weights, g[x]) + bias >=0):
            testerror = testerror+1

if (f.size !=0):
    print(testerror)
    print(testerror/(f.size+g.size))

trainingerror = 0

for x in range(w.size):
    #print (np.matmul(weights, a[x]) + bias)
    if w[x] == 1:
        if (np.matmul(weights, q[x]) + bias <=0):
            trainingerror = trainingerror+1
    if w[x] == 0:
        if (np.matmul(weights, q[x]) + bias >=0):
            trainingerror = trainingerror+1

#can't do this


print(trainingerror)
print(trainingerror/a.size)
print("total error", (testerror+trainingerror)/(f.size+a.size))





