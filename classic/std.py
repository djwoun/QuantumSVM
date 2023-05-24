# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from numpy import genfromtxt


setosa = pd.read_csv('testResult/9'+'.csv')
#genfromtxt('testResult/9'+'.csv', delimiter=',',
       #           skip_header = 0, dtype=None)

#print(setosa.size)
#print(setosa)

print(setosa["Training Error"])