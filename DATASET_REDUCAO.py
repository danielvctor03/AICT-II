#REDUZINDO O TAMANHO DO DATASET

import numpy as np
import pandas as pd


df = np.fromfile(open("D:\Dados capturados\WIFI.npy") , dtype = np.complex64)

k = df.shape[0]

WIFI_REDUZIDO = pd.DataFrame(df[:200000008])

np.save("WIFI_REDUZIDO.npy" , WIFI_REDUZIDO.values)

print(k)

print(WIFI_REDUZIDO)

WIFI_REDUZIDO.to_csv('WIFI_REDUZIDO.csv')












