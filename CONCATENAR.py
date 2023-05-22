#JUNTANDO OS DATASETS EM UM SO

import numpy as np
import pandas as pd

CLEAR = pd.read_csv("D:\Dados capturados\DADOS CSV REDUZIDOS\CLEAR_REDUZIDO.csv")

WIFI = pd.read_csv("D:\Dados capturados\DADOS CSV REDUZIDOS\WIFI_REDUZIDO.csv")

LTE = pd.read_csv("D:\Dados capturados\DADOS CSV REDUZIDOS\LTE_1M_REDUZIDO.csv")

df1 = pd.concat([WIFI , LTE])

