# -*- coding: utf-8 -*-
import pandas as pd

a = pd.read_csv('testResult/syn7.csv')
b = pd.read_csv('testResult/syn22.csv')


df = pd.concat([a,b])

df.to_csv(path_or_buf='testResult/syn7.csv')