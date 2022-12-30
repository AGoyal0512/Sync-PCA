from dynamics import *
import pandas as pd
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from NNetwork import NNetwork as nn
import pickle

def subgraphs_realworld(ntwk, sample_size, k, filename):
    """Generate subgraphs by MCMC sampling from the provided real-world network.

    Args:
        ntwk (string): Input real-world network to the model
        sample_size (int): number of subgraphs to sample
        k (int): number of nodes in each subgraph
        filename (str): output filename

    Returns:
        None
    """
    path = "data/networks/" + str(ntwk) + '.txt'
    G = nn.NNetwork()
    G.load_add_edges(path, increment_weights=False, use_genfromtxt=True)
    X, _ = G.get_patches(k=k, sample_size=sample_size, skip_folded_hom=True)
    pickle.dump(X, open(filename, 'wb'))

def subgraphs_nws(sample_size, k, filename):
    """Generating subgraphs by MCMC sampling from a large NWS network.

    Args:
        sample_size (int): number of subgraphs to sample
        k (int): number of nodes in each subgraph
        filename (str): output filename
    """
    new_nodes = {e: n for n, e in enumerate(NWS.nodes, start=1)} # type: ignore
    new_edges = [(new_nodes[e1], new_nodes[e2]) for e1, e2 in NWS.edges] # type: ignore
    edgelist = []
    for i in range(len(new_edges)):
        temp = [str(new_edges[i][0]), str(new_edges[i][1])]
        edgelist.append(temp)
    G = nn.NNetwork()
    G.add_edges(edgelist)
    X, _ = G.get_patches(k=k, sample_size=sample_size, skip_folded_hom=True)
    pickle.dump(X, open(filename, 'wb'))

ntwks = ['Caltech36', 'UCLA26']
sample_size = 1000
k_vals =  [10, 15, 20, 25, 30]
NWS = nx.newman_watts_strogatz_graph(20000, 1000, 0.67, seed=42)

for ntwk in ntwks:
    print(f"########################\nCurrently on network: {ntwk}.....\n########################")
    for k in k_vals:
        print(f"########################\nCurrently on k= {k}.....\n########################")
        filename = f"output/{ntwk}_{sample_size}_{k}.pkl"
        subgraphs_realworld(ntwk, sample_size, k, filename)
        print(f"########################\nDone.\n########################")


print(f"########################\nCurrently on network: NWS.....\n########################")
for k in k_vals:
    print(f"########################\nCurrently on k= {k}.....\n########################")
    filename = f"output/NWS_{sample_size}_{k}.pkl"
    subgraphs_nws(sample_size, k, filename)
    print(f"########################\nDone.\n########################")