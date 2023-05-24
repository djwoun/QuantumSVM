from sklearn import svm, datasets
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
import numpy as np

iris = datasets.load_iris()

X =  iris.data[:, :2]
y =  iris.target


setosa = X [:50,:2]
versicolor = X [50:100, :2]
virginica = X [100:150, :2]
print(setosa)

def svmFunc(X,y):
    clf = svm.SVC(kernel="linear", C=1000)
    
    #get_params([deep])
    #print (svm.class_weight)
    clf.fit(X, y)
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
    print("weights: ", clf.coef_)
    print("bias: ",clf.intercept_)

q = np.concatenate((setosa, versicolor), axis=0)
w = y[:100]
svmFunc(q,w)



