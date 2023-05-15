
import numpy as np
import pandas as pd


dados = np.load('LTE_FLOOD.bin.npy')

df = pd.DataFrame(dados)

arquivo_CSV = df.to_csv('LTE_FLOOD.csv') 


print(df)
