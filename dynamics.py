import numpy as np
import pandas as pd
import statistics as s
import warnings
from math import floor

import networkx as nx
from NNetwork import NNetwork as nn

warnings.filterwarnings("ignore")

########################
# FCA Dynamics
########################
def width_compute(coloring, kappa):
    """Computes the width of the FCA Dynamics

    Args:
        coloring: Current state in the FCA dynamics
        kappa: k for k-color FCA

    Returns:
        int: The width of the FCA Dynamics
    """
    differences = [np.max(coloring) - np.min(coloring)]
    for j in range(1, kappa+1):
        shifted = (np.array(coloring) + j) % kappa
        differences.append(np.max(shifted) - np.min(shifted))
    return np.min(differences)


def FCA(G, s, k, iteration):
    """Implements the Firefly Cellular Automata model

    Args:
        G (NetworkX Graph): Input graph to the model
        s (array): Initial state
        k (int): k-color FCA
        iteration (int): Number of iterations

    Returns:
        ret: States at each iteration
        label: Whether the system concentrates at the final iteration
    """
    b = (k-1)//2  # Blinking color
    ret = s
    s_next = np.zeros(G.number_of_nodes())
    for h in range(iteration):
        if h != 0:
            s = s_next  # Update to the newest state
            ret = np.vstack((ret, s_next))
        s_next = np.zeros(G.number_of_nodes())
        for i in range(G.number_of_nodes()):
            flag = False  # True if inhibited by the blinking neighbor
            if s[i] > b:
                for j in range(G.number_of_nodes()):
                    if s[j] == b and list(G.nodes)[j] in list(G.adj[list(G.nodes)[i]]):
                        flag = True
                if flag:
                    s_next[i] = s[i]
                else:
                    s_next[i] = (s[i]+1) % k
            else:
                s_next[i] = (s[i]+1) % k

    width = width_compute(ret[-1], k)
    label = False
    if (width < floor(k / 2)):  # half circle concentration
        label = True

    return ret, label