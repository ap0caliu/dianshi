#-*- coding:utf-8 -*-
#!/usr/local/python3.6

"""
trainSet:2000pics
testSet:1000pics
n_class:6
"""

import pandas as pd

classnames = ['OCEAN','MOUNTAIN','LAKE','FARMLAND','DESERT','CITY']
n_class = len(classnames)

df = pd.read_csv('labels.csv',header = None,names = ['filename','label'])
for n in classnames:
    print(n,len(df[df.loc[:,'label']==n]),len(df[df.loc[:,'label']==n])/2000)

'''
OCEAN 456 0.228
MOUNTAIN 463 0.2315
LAKE 172 0.086
FARMLAND 218 0.109
DESERT 636 0.318
CITY 55 0.0275
'''
