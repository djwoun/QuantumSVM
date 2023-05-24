# -*- coding: utf-8 -*-
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import genfromtxt
from sklearn import svm, datasets
#from sklearn.inspection import DecisionBoundaryDisplay

noise = [0,5,10,15,20,25,30]   # number of noisy data sets
splitratio = [0.8,0.7,0.6,0.5] # the training to entire data set ratio
df = pd.DataFrame()            # dataframe to store data 
df3 = pd.DataFrame()           # dataframe to average data 

#BNSCSVFiles/ 20

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
            setosa = genfromtxt(sys.argv[1]+'setosa'+str(m)+'.csv', delimiter=',',
                              skip_header = 0, dtype=None)
            versicolor = genfromtxt(sys.argv[1]+'versicolor'+str(m)+'.csv', delimiter=',',
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
                clf = svm.SVC(kernel="linear", C=1)
                start = time.process_time() 
                clf.fit(X, y)
                end = time.process_time() 
                processTime = (end- start)
                
                
                #print(clf.n_support_)
                #print(clf.probA_)
                #plt.scatter(X[:, 0],X[:, 1] ,c=y, s=30, cmap=plt.cm.Paired)
                '''
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
                '''
                #plt.clf()
                
                #print(clf.dual_coef_)
                
                # return weight bias and time
                return clf.coef_, clf.intercept_, processTime, clf
            
            # number of training data set for set A and B
            SplitRatioMDataA = int(float(SplitRatio) * setosa.size * 0.5)
            SplitRatioMDataB = int(float(SplitRatio) * versicolor.size * 0.5)
            
            # a and b are the training data
            # a and b is concatenated for convience
            a = setosa[:SplitRatioMDataA,:2]
            b = versicolor[:SplitRatioMDataB,:2]
            q = np.concatenate((a, b), axis=0)
            
            # f and g are the testing data
            f = setosa[SplitRatioMDataA:int( setosa.size * 0.5 ),:2]
            g = versicolor[SplitRatioMDataB:int( versicolor.size*0.5),:2]
            
            # c and d stores the training data's feature - concatenate for convinience
            c = np.arange(0, SplitRatioMDataA*0.5,0.5, dtype=int)
            d = np.ones(SplitRatioMDataB )
            w =  np.concatenate((c, d), axis=0)
            
            # number of test data sets
            ReverseSplitRatioMDataA = setosa.size - w.size
            
            # c and d stores the training data's feature
            j = np.zeros(int(ReverseSplitRatioMDataA/2))
            k = np.ones(int(ReverseSplitRatioMDataA/2))
            
            middle = w.size + ReverseSplitRatioMDataA/2
            
            count = 0
            
            
            # create the noisy data sets
            if int(Nois) != int(0):
                for x in h:
                    count = count+1
                    if x<w.size:
                        if (w[int(x)] == 1 ):
                            w[int(x)] = 0
                        elif(w[int(x)] == 0):
                            w[int(x)] = 1
                    elif w.size<=x< middle:
                        if (j[int(x-w.size)] == 0 ):
                            j[int(x-w.size)] = 1
                        elif(j[int(x)] == 1):
                            j[int(x)] = 0
                    elif middle<=x< setosa.size:
                        if(k[int(x-middle)] == 1):
                            k[int(x-middle)] = 0
                        elif(k[int(x)] == 0):
                            k[int(x)] = 1
                    if count >= Noise:
                        break
                
                
            #print(w)  
            #print(j)
            #print(k)
         
            # with the readied data call the SVM function
            # store the weights bias and processing time 
            weights, bias, processTime, clf = svmFunc(q,w)
            a1 =  np.concatenate((f, g), axis=0)     
            b1 =  np.concatenate((j, k), axis=0)      
            # count the number of error in test 
            testerror = 0
            #print(clf.predict(a))
            #print(1-clf.score(a1,b1))
            
            CombinedX = np.concatenate((q,a1),axis=0)
            CombinedY = np.concatenate((w,b1),axis=0)
            MinSum = 0
            
            NumberOfSupportVectors = clf.n_support_[0] +clf.n_support_[1]
            for SV in (clf.support_):
                MinSum += (np.matmul(CombinedX[SV], weights[0])+clf.intercept_)*CombinedY[SV]-1
            objective = 0.5*np.matmul(weights[0],weights[0])- MinSum 
            #objective = weights[0][0]/weights[0][1]
            #clf.coef_[0]/clf.coef_[0][1]
            
            testerror = 1-(clf.score(a1,b1))
            #print(f.size)
            
            trainingerror = 1-(clf.score(q,w))
            
            #print(int(SplitRatio*100))
            # store the data and concatenate to dataframe
            df2 = pd.DataFrame({'Training': int(SplitRatio*100),
                                'Noise': Noise,
                                'Training Error': (trainingerror),
                                'Testing Error': (testerror),
                                'Time Count' : processTime,
                                'Weight' : [weights],
                                'Bias' : bias,
                                'Objective Function' : objective,
                                "Number of SV":NumberOfSupportVectors,
                                })
            
            df = pd.concat([df, df2])
            #df4 = pd.concat([df4, df2])
        
df.to_csv(path_or_buf='testResult/9.csv')
#df.to_csv(path_or_buf='testResult/2avg.csv')



