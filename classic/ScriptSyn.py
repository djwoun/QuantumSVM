# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import genfromtxt
from sklearn import svm, datasets
import dask.dataframe
#from sklearn.inspection import DecisionBoundaryDisplay

noise = [0]
#[0,5,10,15,20,25,30]   # number of noisy data sets
splitratio = [0.8,0.7,0.6,0.5] # the training to entire data set ratio
features = [2]#2 ** np.arange(21)
#eatures = np.delete(features,0)


df = pd.DataFrame()            # dataframe to xstore data 
df3 = pd.DataFrame()           # dataframe to average data 

#BNSCSVFiles/ 20

# Essentially the program runs in three for loops
# One loop changes the training to data ratio
# Another loop changes the amount of noise put into data
# The last loop simply reiterates the test information for accuracy

START = time.time()
#print("A",time.process_time() )
for feat in features:        
    for m in range (int(sys.argv[2])):
        for SplRat in splitratio:
            for Nois in noise:
                #print(features)
                csvFile = sys.argv[1] # CSVFile will save the prefix for the csv file location
                #plitRatio = SplRat   # SplitRatio holds the current train/data
                #SplitRatio2 = 1-float(SplitRatio) # SplitRatio holds the current test/data
                #Noise = Nois          # current number of noisy data points
                
                # csv file read
                print("AT: ", m, " FEAT: ", feat)
                print("AB",time.process_time() )
                '''
                setosa = genfromtxt(sys.argv[1]+'X'+str(m)+' '+str(feat)+'.csv', delimiter=',',
                                  skip_header = 0, dtype=None)
                versicolor = genfromtxt(sys.argv[1]+'Y'+str(m)+' '+str(feat)+'.csv', delimiter=',',
                                  skip_header = 0, dtype=None)
                
                # random numbers used for noise read from csv - stored as h
                h = genfromtxt(sys.argv[1]+'random'+str(m)+' '+str(feat)+'.csv', delimiter=',',
                                  skip_header = 0, dtype=None)
                
                setosa, versicolor = datasets.make_blobs(n_samples=256, centers=2, n_features=feat,cluster_std=0.1,
                                           random_state=m,center_box=(0,10))
                np.savetxt('SynCsvFile/X'+str(m)+' '+str(feat)+'.csv', setosa, delimiter=',') 
                np.savetxt('SynCsvFile/Y'+str(m)+' '+str(feat)+'.csv', versicolor, delimiter=',') 
                '''
                setosa, versicolor = datasets.make_blobs(n_samples=256, centers=2, n_features=feat,cluster_std=3,
                                           random_state=m,center_box=(0,10))
                setosa = setosa.astype(int)
                print("B",time.process_time() )
               
                
                def svmFunc(X,y):
                    
                    # specify the kernel
                    # fit(x,y) trains the data set - start and end stores the times
                    # processtime = time it took to run fit
                    clf = svm.SVC(kernel="linear", C=1000)
                    start = time.process_time() 
                    clf.fit(X, y)
                    end = time.process_time() 
                    processTime = (end- start)
                    #print(processTime)
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
                    #plt.savefig('testPlots/Tra'+str(SplRat*100)+'Nois'+str(Nois)+'Run'+str(m)+'.png')
                    plt.clf()
                    '''
                    #print(clf.dual_coef_)
                    
                    # return weight bias and time
                    return clf.coef_, clf.intercept_, processTime, clf
                
                # number of training data set for set A and B
                DivFeat = 1/feat
                SplitRatioMDataA = int(float(SplRat) * setosa.size * (DivFeat))
                #SplitRatioMDataB = int(float(SplitRatio) * versicolor.size * 0.5)
                #print("B",time.process_time() )
                q = setosa[:SplitRatioMDataA,:]
                w = versicolor[:SplitRatioMDataA]
                ReverseSplitRatioMDataA = setosa.size - w.size
                a1 = setosa[SplitRatioMDataA:int( setosa.size * (DivFeat) ),:]
                b1 = versicolor[SplitRatioMDataA:]
                #print("C",time.process_time() )
                '''
                # a and b are the training data
                # a and b is concatenated for convience
                
                #a = setosa[:SplitRatioMDataA,:2]
                #b = versicolor[:SplitRatioMDataB,:2]
                q = np.concatenate((a, b), axis=0)
                
                # f and g are the testing data
                
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
                '''
                middle = w.size + ReverseSplitRatioMDataA/2
                
                count = 0
                
                #print("D",time.process_time() )
                Noise = 0
                # create the noisy data sets
                if int(Noise) != int(0):
                    for x in h:
                        count = count+1
                        if x<SplitRatioMDataA:
                            if (w[int(x)] == 1 ):
                                w[int(x)] = 0
                            elif(w[int(x)] == 0):
                                w[int(x)] = 1
                        elif SplitRatioMDataA<=x< setosa.size:
                            if (b1[int(x-SplitRatioMDataA)] == 0 ):
                                b1[int(x-SplitRatioMDataA)] = 1
                            elif(b1[int(x-SplitRatioMDataA)] == 1):
                                b1[int(x-SplitRatioMDataA)] = 0
                        if count >= Noise:
                            break
                    
                #print("V",time.process_time() )
                 
                # with the readied data call the SVM function
                # store the weights bias and processing time 
                weights, bias, processTime, clf = svmFunc(q,w)
                
                #print("W",time.process_time() )
                '''
                a1 =  np.concatenate((f, g), axis=0)     
                b1 =  np.concatenate((j, k), axis=0)      
                # count the number of error in test 
                testerror = 0
                #print(clf.predict(a))
                #print(1-clf.score(a1,b1))
                for x in range(int(j.size)):
                    #print (np.matmul(weights, a[x]) + bias)
                    if j[x] == 1:
                        if (np.matmul(weights, f[x]) + bias <=0):
                            testerror = testerror+1
                    elif j[x] == 0:
                        if (np.matmul(weights, f[x]) + bias >=0):
                            testerror = testerror+1
                
                # count the number of error in test 
                for x in range(int(g.size*0.5)):
                    #print (np.matmul(weights, b[x]) + bias)
                    if k[x] == 1:
                        if (np.matmul(weights, g[x]) + bias <=0):
                            testerror = testerror+1
                    elif k[x] == 0:
                        if (np.matmul(weights, g[x]) + bias >=0):
                            testerror = testerror+1
                
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
                # calculate the objective function (1/2 W (dot) W)
                #weightsT = np.transpose(weights)
                #print(weightsT)
                #objective = np.matmul(weights, weightsT)
                #print(objective)
                #print("X",time.process_time() )
                trainingerror = 1 - clf.score(q,w)
                testingerror = (1-clf.score(a1,b1))
                #objective = 0.5*np.dot(weights[0],weights[0])
                CombinedX = np.concatenate((q,a1),axis=0)
                CombinedY = np.concatenate((w,b1),axis=0)
                MinSum = 0
                #print("Y",time.process_time() )
                NumberOfSupportVectors = clf.n_support_[0] +clf.n_support_[1]
                for SV in (clf.support_):
                    MinSum += (np.dot(CombinedX[SV], weights[0])+clf.intercept_)*CombinedY[SV]-1
                objective = 0.5*np.dot(weights[0],weights[0])- MinSum 
                #print(objective)
                #print(int(SplitRatio*100))
                print("weights: ", weights)
                # store the data and concatenate to dataframe
                df2 = pd.DataFrame({
                                    'Training': int(SplRat*100),
                                    'Noise': Nois,
                                    'Training Error': trainingerror,
                                    'Testing Error': testingerror,
                                    'Time Count' : processTime,
                                    #'Weight' : [weights],
                                    #'Bias' : bias,
                                    'Objective Function' : objective,
                                    "Number of SV":NumberOfSupportVectors,
                                    "Features": feat,
                                    },index = [0])
                
                df = pd.concat([df, df2])
                print("Z",time.process_time() )
            
#df.to_csv(path_or_buf='testResult/Wegight.csv')
#df.to_csv(path_or_buf='testResult/2avg.csv')


END = time.time()

print( END  - START)