# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

from sklearn import svm, datasets
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.pyplot as plt
import numpy as np
import sys

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


def svmFunc(X,y):
    clf = svm.SVC(kernel="linear", C=1000)
    
    #get_params([deep])
    #print (svm.class_weight)
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

SplitRatioMDataA = int(float(SplitRatio) * virginica.size * 0.5)
SplitRatioMDataB = int(float(SplitRatio) * versicolor.size * 0.5)
#print(setosa.size)
#print(int(float(SplitRatio) * setosa.size))
a = virginica[:SplitRatioMDataA,:2]
#print(a)
#print(a.size)

#print(setosa[:int(float(SplitRatio) * setosa.size * 0.5),:2])
#print(d)
b = versicolor[:SplitRatioMDataB,:2]
q = np.concatenate((a, b), axis=0)
#c = y[:SplitRatioMDataA]
#d = y[int(setosa.size*0.5):int(setosa.size*0.5)+int(float(SplitRatio) * setosa.size*0.5)]
c = np.arange(0, SplitRatioMDataA*0.5,0.5, dtype=int)
d = np.ones(SplitRatioMDataB )


w =  np.concatenate((c, d), axis=0)
weights, bias = svmFunc(q,w)



f = virginica[SplitRatioMDataA:int( virginica.size * 0.5 ),:2]
g = versicolor[SplitRatioMDataB:int( versicolor.size*0.5),:2]

print(f.size)

testerror = 0
for x in range(int(f.size*0.5)):
    #print (np.matmul(weights, f[x]) + bias)
    if (np.matmul(weights, f[x]) + bias >0):
        testerror = testerror+1
#print(j)
#print(h[0])
for x in range(int(g.size*0.5)):
    #print (np.matmul(weights, g[x]) + bias)
    if (np.matmul(weights, g[x]) + bias < 0):
        testerror = testerror+1


print(testerror)
print(testerror/f.size)


trainingerror = 0
for x in range(int(a.size*0.5)):
    #print (np.matmul(weights, f[x]) + bias)
    if (np.matmul(weights, a[x]) + bias >0):
        trainingerror = trainingerror+1
#print(j)
#print(h[0])
for x in range(int(b.size*0.5)):
    #print (np.matmul(weights, g[x]) + bias)
    if (np.matmul(weights, b[x]) + bias < 0):
        trainingerror = trainingerror+1


print(trainingerror)
print(trainingerror/a.size)
print("total error", (testerror+trainingerror)/(f.size+a.size))



