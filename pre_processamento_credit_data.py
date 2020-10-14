# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 11:12:39 2020

@author: denis
"""


import pandas as pd
import numpy as np
base = pd.read_csv('credit_data.csv')
base.describe()
base.loc[base['age']<0]
base['age'][base.age>0].mean() # média das idades sem os valores negativos
base.loc [base.age<0,'age']= 40.92
# resgistros com problemas
base.loc[pd.isnull(base['age'])] # mostra apenas os valores que estão com elementos nulos
previsores = base.iloc[:,1:4].values #variavel 

classe = base.iloc[:,4]


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy='mean')
imputer = imputer.fit(previsores[:,0:3])
previsores [:, 0:3] = imputer.transform(previsores[:,0:3])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)