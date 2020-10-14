# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 09:19:01 2020

@author: denis
"""


import pandas as pd

base = pd.read_csv('census.csv')
previsores = base.iloc[:,0:14].values
classe = base.iloc[:,14].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_previsores = LabelEncoder()
#labels = labelencoder_previsores.fit_transform(previsores[:,1]) # Transformação de variáveis categóricas I - base censo
labels = labelencoder_previsores.fit_transform(previsores[:,3])
labels = labelencoder_previsores.fit_transform(previsores[:,5])
labels = labelencoder_previsores.fit_transform(previsores[:,6])
labels = labelencoder_previsores.fit_transform(previsores[:,7])
labels = labelencoder_previsores.fit_transform(previsores[:,8])
labels = labelencoder_previsores.fit_transform(previsores[:,9])
labels = labelencoder_previsores.fit_transform(previsores[:,13])

onehotencorder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [1,3,5,6,7,8,9,13])],remainder='passthrough')
previsores = onehotencorder.fit_transform(previsores).toarray()

labelencorder_classe = LabelEncoder()
classe = labelencorder_classe.fit_transform(classe)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)












