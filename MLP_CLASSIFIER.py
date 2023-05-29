#BIBLIOTECAS

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


#CARREGAMENTO DOS DATASETS

D_CLEAR = np.load("D:\Dados capturados\DATASETS\CLEAR.npy")
D_WIFI = np.load("D:\Dados capturados\DATASETS\WIFI.npy")
D_LTE = np.load("D:\Dados capturados\DATASETS\LTE_1M.npy")

#REDUCAO DO DATASET

DF1 = D_CLEAR[:70000000]
DF2 = D_WIFI[:70000000]
DF3 = D_LTE[:70000000]

#SEPARACAO DAS PARTES REAL E IMAGINARIA
def sep_col_comp(dados):
    parte_real = dados.real
    parte_imag = dados.imag
    return parte_real , parte_imag

def montagem_datasets(dados_1 , dados_2):
    real1 , imag1 = sep_col_comp(dados_1)
    real2 , imag2 = sep_col_comp(dados_2)
    atrib1 = np.ones(len(dados_1))
    atrib2 = np.ones(len(dados_2))
    dados1 = np.column_stack((real1,imag1,atrib1))
    dados2 = np.column_stack((real2,imag2,atrib2))
    conjunto = np.vstack([dados1 , dados2])
    indices = np.random.permutation(len(conjunto))
    dados = conjunto[indices]
    return dados

def remodel(dados , feature):
    df = pd.DataFrame(dados , columns = ['real' , 'imag' , 'clfq'])
    df_junt = df[['real' , 'imag']].to_numpy()
    dados_dim = df_junt.reshape(len(dados)//100 , 100 , feature)
    dados_dim = dados_dim.astype(np.float32)
    alvo = df['clfq'].values
    alvo_dim = alvo.reshape(len(alvo)//100 , 100 , feature-1)
    alvo_dim = alvo_dim.astype(np.float32)
    return dados_dim , alvo_dim

def processos(dados_1 , dados_2 , feature):
    dados = montagem_datasets(dados_1 , dados_2)
    X , Y = remodel(dados , feature)
    return X , Y


#CLEAR X WIFI

X , Y = processos(DF1 , DF2 , 2)

#REMODELANDO PARA UM ARRAY 2D

X = X.reshape(X.shape[0],-1)
Y = Y.reshape(Y.shape[0],-1)


#CONJUNTO DE TREINAMENTO E TESTE

X_train , X_test , Y_train , Y_test = train_test_split(X , Y , test_size=0.45 , random_state=0)


#NORMALIZACAO DOS DADOS

mm = MinMaxScaler()

X_train = mm.fit_transform(X_train)
X_test = mm.transform(X_test)

#ESTRUTURA DA REDE NEURAL
classifier = tf.keras.models.Sequential()

classifier.add( tf.keras.layers.Dense( activation="relu" , input_dim=200 ,               
                                       units=500 , kernel_initializer='uniform'  ) )

classifier.add( tf.keras.layers.Dense( 100 , activation='sigmoid' ))

classifier.compile( optimizer='adam' , loss='binary_crossentropy' , metrics=['accuracy'])

#TREINAMENTO DO MODELO

classifier.fit(X_train , Y_train , batch_size=15 , epochs=81)


##############################################################################################################

#CLEAR X LTE

X , Y = processos(DF1 , DF3 , 2)

#REMODELANDO PARA UM ARRAY 2D

X = X.reshape(X.shape[0],-1)
Y = Y.reshape(Y.shape[0],-1)


#CONJUNTO DE TREINAMENTO E TESTE

X_train , X_test , Y_train , Y_test = train_test_split(X , Y , test_size=0.45 , random_state=0)


#NORMALIZACAO DOS DADOS

mm = MinMaxScaler()

X_train = mm.fit_transform(X_train)
X_test = mm.transform(X_test)

#ESTRUTURA DA REDE NEURAL
classifier = tf.keras.models.Sequential()

classifier.add( tf.keras.layers.Dense( activation="relu" , input_dim=200 ,               
                                       units=500 , kernel_initializer='uniform'  ) )

classifier.add( tf.keras.layers.Dense( 100 , activation='sigmoid' ))

classifier.compile( optimizer='adam' , loss='binary_crossentropy' , metrics=['accuracy'])

#TREINAMENTO DO MODELO

classifier.fit(X_train , Y_train , batch_size=15 , epochs=81)


####################################################################################################

#WIFI X LTE

X , Y = processos(DF2 , DF3 , 2)

#REMODELANDO PARA UM ARRAY 2D

X = X.reshape(X.shape[0],-1)
Y = Y.reshape(Y.shape[0],-1)


#CONJUNTO DE TREINAMENTO E TESTE

X_train , X_test , Y_train , Y_test = train_test_split(X , Y , test_size=0.45 , random_state=0)


#NORMALIZACAO DOS DADOS

mm = MinMaxScaler()

X_train = mm.fit_transform(X_train)
X_test = mm.transform(X_test)

#ESTRUTURA DA REDE NEURAL
classifier = tf.keras.models.Sequential()

classifier.add( tf.keras.layers.Dense( activation="relu" , input_dim=200 ,               
                                       units=500 , kernel_initializer='uniform'  ) )

classifier.add( tf.keras.layers.Dense( 100 , activation='sigmoid' ))

classifier.compile( optimizer='adam' , loss='binary_crossentropy' , metrics=['accuracy'])

#TREINAMENTO DO MODELO

classifier.fit(X_train , Y_train , batch_size=15 , epochs=81)

