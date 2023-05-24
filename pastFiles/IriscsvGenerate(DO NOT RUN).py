# -*- coding: utf-8 -*-
import csv
from sklearn import datasets
import numpy as np
import random
iris = datasets.load_iris()

X =  iris.data[:, :2]
y =  iris.target



virginica = X [100:150, :2]



for x in range (30):
    setosa = X [:50,:2]
    versicolor = X [50:100, :2]
   
    np.random.shuffle(setosa)
    np.random.shuffle(versicolor)
    
    ran = random.sample(range(100), int(30))
    
    
    np.savetxt('IrisCSVFiles/setosa'+str(x)+'.csv', setosa, delimiter=',') 
    np.savetxt('IrisCSVFiles/versicolor'+str(x)+'.csv', versicolor, delimiter=',')
    np.savetxt('IrisCSVFiles/random'+str(x)+'.csv', ran, delimiter=',')
   
    '''
    with open('IrisCSVFiles/test'+str(x)+'.csv', "w") as file:
        
        spamwriter = csv.writer(file, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_NONE)
        spamwriter.writerow([(str(1.1112233), str(1.3355)),(str(1.1112233), str(1.3355))])
    '''
        #fieldnames = ['setosa','versicolor']
        #theWriter = csv.DictWriter(f,fieldnames=fieldnames)
        #theWriter.writeheader()
        #for i in range(50):
         #   theWriter.writerow({'setosa':setosa[i],'versicolor':versicolor[i]})
