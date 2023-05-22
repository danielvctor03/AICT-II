
import numpy as np
import pandas as pd


dados = np.load('LTE_1M.npy')

df = pd.DataFrame(dados)

arquivo_CSV = df.to_csv('LTE_1M.csv')

print(df)













