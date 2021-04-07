import os
import numpy as np
import pandas as pd
import scipy

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm

import networkx as nx


def plot_network_centrality(g, centrality_func=nx.information_centrality):
    """A function to plot a network with some measure of 
    centrality as the node color"""

    centralities = centrality_func(g, weight='weight')
    centralities = list(centralities.values())
    edge_weights = [g[u][v]['weight'] for u,v in g.edges()]
    pos = nx.spring_layout(g, weight='weight')
    nx.draw_networkx(g, 
                     pos, 
                     with_labels = False, 
                     node_color = centralities, 
                     edgecolors = 'black',
                     width = edge_weights,
                     edge_cmap = plt.get_cmap('Greys_r'),
                     edge_color = edge_weights,
                     cmap = 'Spectral', 
                     alpha = 0.7) 
    
    sm = plt.cm.ScalarMappable(cmap='Spectral')
    sm._A = []
    plt.colorbar(sm)
