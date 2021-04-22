import os
import numpy as np
import pandas as pd
import scipy
from scipy.cluster.hierarchy import dendrogram

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


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)