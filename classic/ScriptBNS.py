# -*- coding: utf-8 -*-
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import genfromtxt
from sklearn import svm, datasets
from sklearn.inspection import DecisionBoundaryDisplay

noise = [0,5,10,15,20,25,30]   # number of noisy data sets
splitratio = [0.8,0.7,0.6,0.5] # the training to entire data set ratio
df = pd.DataFrame()            # dataframe to store data 
df3 = pd.DataFrame()           # dataframe to average data 

#BNSCSVFiles/ 30

# Essentially the program runs in three for loops
# One loop changes the training to data ratio
# Another loop changes the amount of noise put into data
# The last loop simply reiterates the test information for accuracy
for SplRat in splitratio:
    for Nois in noise:
        df4 = pd.DataFrame()            # dataframe to temorarily store the current data
        for m in range (int(sys.argv[2])):
            
            csvFile = sys.argv[1] # CSVFile will save the prefix for the csv file location
            SplitRatio = SplRat   # SplitRatio holds the current train/data
            SplitRatio2 = 1-float(SplitRatio) # SplitRatio holds the current test/data
            Noise = Nois          # current number of noisy data points
            
            # csv file read
            setosa = genfromtxt(sys.argv[1]+'BN'+str(m)+'.csv', delimiter=',',
                              skip_header = 0, dtype=None)
            versicolor = genfromtxt(sys.argv[1]+'NS'+str(m)+'.csv', delimiter=',',
                              skip_header = 0, dtype=None)
            
            # random numbers used for noise read from csv - stored as h
            h = genfromtxt(sys.argv[1]+'random'+str(m)+'.csv', delimiter=',',
                              skip_header = 0, dtype=None)
            
            # store the iris data set 0 to 1 (this is strictly for convinience)
            iris = datasets.load_iris()
            y =  iris.target
            
            def svmFunc(X,y):
                
                # specify the kernel
                # fit(x,y) trains the data set - start and end stores the times
                # processtime = time it took to run fit
                clf = svm.SVC(kernel="poly", C=1,degree=3)
                #clf = svm.SVC(kernel="rbf", C=1)
                start = time.process_time() 
                clf.fit(X, y)
                end = time.process_time() 
                processTime = (end- start)
                
                 
                '''
                
                plt.scatter(X[:, 0],X[:, 1] ,c=y, s=30, cmap=plt.cm.Paired)
                
            # plot the decision function
                ax = plt.gca()
                
                #plt.plot(x, clf.coef_[0]/clf.coef_[0][1])*x-clf.intercept_/clf.coef_[0][1], '-r', label='y=2x+1')
                
                
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
                plt.savefig('testPlots/Tra'+str(SplRat*100)+'Nois'+str(Nois)+'Run'+str(m)+'.png')
                plt.clf()
                
                #print(clf.dual_coef_)
                '''
                # return weight bias and time
                return processTime, clf
            
            # number of training data set for set A and B
            SplitRatioMDataA = int(float(SplitRatio) * setosa.size * (1/9))
            SplitRatioMDataB = int(float(SplitRatio) * versicolor.size * (1/9))
            

            # a and b are the training data
            # a and b is concatenated for convience
            a = setosa[:SplitRatioMDataA,:9]
            b = versicolor[:SplitRatioMDataB,:9]
            q = np.concatenate((a, b), axis=0)
            
            
            # f and g are the testing data
            f = setosa[SplitRatioMDataA:int(setosa.size * (1/9)),:9]
            g = versicolor[SplitRatioMDataA:int(setosa.size * (1/9)),:9]
           
            # c and d stores the training data's feature - concatenate for convinience
            c = (np.ones(int(50*SplitRatio)))
            d = (-1*np.ones(int(50*SplitRatio) ))
            w =  np.concatenate((c, d), axis=0)
            
            # number of test data sets
            ReverseSplitRatioMDataA = int(setosa.size * (1/9 ) - SplitRatioMDataA)
            
            # c and d stores the training data's feature
            j = np.ones(int(ReverseSplitRatioMDataA))
            k = -1*np.ones(int(ReverseSplitRatioMDataA))
           
            middle = SplitRatioMDataA*2 + ReverseSplitRatioMDataA
            
            count = 0
            
            
            #print(SplitRatioMDataA*2)
            # create the noisy data sets
            #print(h)
            #print(middle)
            if int(Nois) != int(0):
                for x in h:
                    count = count+1
                    if x<(SplitRatioMDataA*2):
                        if (int(w[int(x)]) == 1 ):
                            w[int(x)] = -1
                        elif(int(w[int(x)])== int(-1) ):
                            w[int(x)] = 1
                    elif (SplitRatioMDataA*2)<=x< middle:
                        if (int(j[int(x-SplitRatioMDataA*2)]) == int(-1) ):
                            j[int(x-SplitRatioMDataA*2)] = 1
                        elif(int(j[int(x-SplitRatioMDataA*2)]) ==1 ):
                            j[int(x-SplitRatioMDataA*2)] = -1
                    elif middle<=x< 100:
                        if(int(k[int(x-middle)]) == 1):
                            k[int(x-middle)] = -1
                        elif(int(k[int(x-middle)]) == int(-1)):
                            k[int(x-middle)] = 1
                    if count >= Noise:
                        break
            
           
            # print(w)
            # with the readied data call the SVM function
            # store the weights bias and processing time 
            processTime, clf = svmFunc(q,w)
            a1 =  np.concatenate((f, g), axis=0)     
            b1 =  np.concatenate((j, k), axis=0)      
            # count the number of error in test 
            testerror = 1-(clf.score(a1,b1))
            
            
            trainingerror = 1-(clf.score(q,w))
            
            '''
            # count the number of error in training 
            trainingerror = 0
            for x in range(w.size):
                #print (np.matmul(weights, a[x]) + bias)
                if w[x] == 1:
                    if (np.matmul(weights, q[x]) + bias <=0):
                        trainingerror = trainingerror+1
                elif w[x] == 0:
                    if (np.matmul(weights, q[x]) + bias >=0):
                        trainingerror = trainingerror+1
            '''
            #print(SplitRatio)
            # store the data and concatenate to dataframe
            df2 = pd.DataFrame({'Training': int(SplitRatio*100),
                                'Noise': Noise,
                                'Training Error': trainingerror,
                                'Testing Error': (testerror),
                                'Time Count' : processTime,
                                },index=[0])
            
            df = pd.concat([df, df2])
            df4 = pd.concat([df4, df2])
        
#df.to_csv(path_or_buf='testResult/2BAS.csv')
#df.to_csv(path_or_buf='testResult/2avg.csv')



