import random
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

import networkx as nx
from dynamics import *

from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

ntwk = "Caltech36"
sample_size = 1000
k = 30
X = pd.read_pickle(f"output/{ntwk}_{sample_size}_{k}.pkl")
X = X.T

x = []
y = []
for i in range(X.shape[0]):
    G = nx.from_pandas_adjacency(np.array(X[0][i]).reshape(k, k))
    ret = FCA(G, s = np.random.randint(1, 6), k=5, iteration=50)
    x.append(ret[0])
    y.append(ret[1])

print(f"Shape of x: {np.array(x).shape}")
print(f"Shape of y: {len(y)}")