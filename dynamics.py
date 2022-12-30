import numpy as np
import warnings
from math import floor

import networkx as nx

warnings.filterwarnings("ignore")

#####################################
# Firefly Cellular Automata Dynamics
#####################################
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
    if (width < floor(k / 2)):  # Half circle concentration
        label = True

    return ret, label


#####################################
# Greenberg-Hasting Model Dynamics
#####################################
def GHM(G, s, k, iteration):
    """Implements the Greenberg-Hastings model

    Args:
        G (NetworkX Graph): Input graph to the model
        s (array): Initial state
        k (int): k-color GHM
        iteration (int): Number of iterations

    Returns:
        ret: States at each iteration
        label: Whether the system concentrates at the final iteration
    """
    ret = s
    s_next = np.zeros(G.number_of_nodes())

    for h in range(iteration):
        if h != 0:
            s = s_next  # Update to the newest state
            ret = np.vstack((ret, s_next))
        s_next = np.zeros(G.number_of_nodes())
        for i in range(G.number_of_nodes()):
            if s[i] == 0:
                flag = True  # If coloring of neighbor of 1 is not found
                for j in range(G.number_of_nodes()):
                    if s[j] == 1 and list(G.nodes)[j] in list(G.adj[list(G.nodes)[i]]):
                        s_next[i] = 1
                        flag = False
                        break
                if flag:
                    s_next[i] = 0
            else:
                s_next[i] = (s[i]+1) % k

    label = False
    if np.sum(ret[-1]) == 0:
        label = True

    return ret, label


#####################################
# Kuramoto Model Dynamics
#####################################
def Kuramoto(G, K, s, iteration, step):
    """Implements the Kuramoto model

    Args:
        G (NetworkX Graph): Input graph to the model
        s (array): Initial state
        k (int): k-color GHM
        iteration (int): Number of iterations
        step (float): Step size for Kuramoto discretization

    Returns:
        ret: States at each iteration
        label: Whether the system concentrates at the final iteration
    """

    ret = s
    s_next = np.zeros(G.number_of_nodes())
    for h in range(iteration-1):
        if h != 0:
            s = s_next  # Update to the newest state
            ret = np.vstack((ret, s_next))
        for i in range(G.number_of_nodes()):
            neighbor_col = []
            for j in range(G.number_of_nodes()):
                if list(G.nodes)[j] in list(G.adj[list(G.nodes)[i]]):
                    neighbor_col.append(s[j])

            new_col = s[i] + step * K * np.sum(np.sin(neighbor_col - s[i]))
            if np.abs(new_col) > np.pi:
                if new_col > np.pi:
                    new_col -= 2*np.pi
                if new_col < -np.pi:
                    new_col += 2*np.pi
            s_next[i] = new_col

    label = False
    if widthkura(ret[-1]) < np.pi:
        label = True

    return ret, label


def widthkura(colors):
    """Computes the width of the Kuramoto Dynamics

    Args:
        colors: Current state in the Kuramoto dynamics

    Returns:
        int: The width of the Kuramoto Dynamics
    """
    ordered = list(np.pi - colors)
    ordered.sort()
    lordered = len(ordered)
    threshold = np.pi

    if lordered == 1:
        return 0

    elif lordered == 2:
        dw = ordered[1]-ordered[0]
        if dw > threshold:
            return 2*np.pi - dw
        else:
            return dw

    else:
        widths = [2*np.pi+ordered[0]-ordered[-1]]
        for i in range(lordered-1):
            widths.append(ordered[i+1]-ordered[i])
        return np.abs(2*np.pi - max(widths))