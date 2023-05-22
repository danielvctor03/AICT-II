#METODO PERCEPTRON DE MULTIPLAS CAMADAS (MLP)
#ESSE METODO SERA UTILIZADO PARA REALIZAR METRICAS E COMPARAR TRES AMOSTRAS DE SINAIS (WIFI, LTE E RUIDO)

#BIBLIOTECAS BASICAS
import numpy as np
import pandas as pd

#BIBLIOTECA DE PLOTAGEM DE GRAFICOS
import matplotlib.pyplot as plt

#BIBLIOTECAS DE MACHINE LEARNING
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split #TREINAMENTO DO DATASET
from sklearn.metrics import r2_score #EFICIENCIA DO MODELO
from sklearn.linear_model import SGDRegressor  
from sklearn.neural_network import MLPRegressor

#CARREGAMENTO DOS DATASETS

dataset1 = pd.read_csv("D:\Dados capturados\DADOS CSV REDUZIDOS\CLEAR_REDUZIDO.csv")





















