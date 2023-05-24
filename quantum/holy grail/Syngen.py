# -*- coding: utf-8 -*-
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import genfromtxt
from sklearn import svm, datasets
import random
import pyqubo as pq
from pyqubo import Spin
from pyqubo import Array
from sklearn.inspection import DecisionBoundaryDisplay
import neal

standardDeviation = [3]
features = [2]
#[2 ** np.arange(21)]
#features = np.delete(features,0)
def go_fast(p,X,y):
    N = X.shape[0] 
    P = np.kron(np.eye(N), p)
    A = 0.5 * P.T @ (np.multiply(X@X.T, np.outer(y, y))) @ P
    b = - P.T @ np.ones(N)
    
    return A, b

def weightGen(s):
    return np.random.choice(np.arange(0.5,4,0.5),size=s, replace=True)

def NweightGen(s):
    return -np.random.choice(np.arange(0.5,4,0.5),size=s, replace=True)

for feat in features:
    for std in standardDeviation:
        for x in range(1):
            N=256
            
            X = 10*(np.random.rand(600,feat)-(0.5 ))
            
            weight = weightGen(int(feat))
            
            bias = weightGen(1)
            
            
            print(weight)
            
            #f=0
            #if feat == 2:
            #    f = 1.1
            
            #print(weight)
            #print(bias)
            #print(X[0])
            #print (np.dot(X[0],weight)+1+bias)
            #print(np.dot(X[0],weight)-1+bias)
           
            a = np.empty(feat,dtype=int)
            b = np.empty(0,dtype=int)
            d = np.empty(feat,dtype=int)
            e = np.empty(0,dtype=int)
            
           
            for i in range(500):
                #print(np.dot(X[i],weight)+bias)
                #if (d.size+a.size)>feat*(N+1):
                #    break
                if ((np.dot(X[i],weight)+bias <=-(2) and (a.size)<=feat*((N+1)/2)) ):
                    a=np.vstack((a,X[i]))
                    b = np.append(b,-1)
                    #print("A")
                if ((np.dot(X[i],weight)+bias >=(2)) and (d.size)<=feat*((N+1)/2) ):
                    
                    d=np.vstack((d,X[i]))
                    e = np.append(e,1)
                    #print("B")
            
            #print(weight)
            
            a = np.delete(a,0,axis=0)
            d = np.delete(d,0,axis=0)
            
            
            #print(weight)
            #print(len(a))
            #print(len(d))
            if(len(a)<=1 or len(d)<=1):
                break
            
           
            
            
            a = np.concatenate((a,d))
            b = np.concatenate((b,e))
            print(a.size/feat)
            
            
         
            #a = np.array([[1,0.25], [1,0.25]])
            #b = [0,1]
            clf = svm.SVC(kernel="linear", C=10000)
            start = time.process_time() 
            clf.fit(a, b)
            end = time.process_time() 
            print("WEIGHT : " ,clf.coef_,"BIAS : " ,clf.intercept_)
            print("FEATURE", feat, "SCORE: " ,clf.score(a,b))
            processTime = (end- start)
            #print( clf.coef_)
            #print(clf.intercept_)
            
            
            plt.scatter(a[:, 0], a[:, 1], c=b, s=30, cmap=plt.cm.Paired)

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
            
            
            
            
            
            
            
            
            
            #s = Array.create('s', shape=4, vartype='SPIN')
            #H = sum(n * s for s, n in zip(s, numbers))**2
            #print(H)
            p = np.array([0.5,1,2])
            f,g = go_fast(p,a,b) 
            
            #s1= Spin("s1")
            #s1 = np.repeat(s1, repeats=180, axis=0)
            s1 = Array.create('b', shape=N*3, vartype='BINARY')
            
         
            #print(s1.shape)
            #print(f.shape)
            H = Array.dot (Array.dot (s1.T,f),s1 ) +Array.dot(s1.T,Array(g))
            #print(Array.dot(s1.T,Array(g)))
            #print(dir(H))
            model = H.compile()
            #print(dir(model))
            #print(H,"\n")
            
            bqm = model.to_bqm()

            sa = neal.SimulatedAnnealingSampler()
            sampleset = sa.sample(bqm, num_reads=10)
            decoded_samples = model.decode_sampleset(sampleset)
            best_sample = min(decoded_samples, key=lambda x: x.energy)
           
            #print(best_sample.sample,"\n")
            
            
            #print(best_sample.sample.values())
            answer = np.array(list(best_sample.sample.values()))
            print(answer)
            #print(list(best_sample.sample.values()).index(1))
            #print(list(best_sample.sample.keys())[list(best_sample.sample.values()).index(1)])
            P = np.kron(np.eye(a.shape[0]), p)
            lmda = np.inner(answer,P)
            
            print(lmda)
            weight=0
            
                
            var1 = lmda * b
            var2 = (a.T * var1).T
            w = np.sum(var2, axis=0)
            print(w)
            
            