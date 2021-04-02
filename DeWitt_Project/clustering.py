"""
Notebook to cluster V sequences from a donor/file type
"""


# %%
# --------------------------------------------------
""" Imports """

import os
import numpy as np
import pandas as pd
import scipy
from itertools import combinations
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from importlib import reload
import networkx as nx
import Levenshtein 

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm

# local imports
import file_loader_funcs as _load
import distance_funcs as _dist

%matplotlib inline


# %%
# --------------------------------------------------
""" Define a sample """

reload(_load)

ROOT_DIR = "/Volumes/Cooper_TB_Drive/research/rajapakse/b_cell_1/public-bcell-dataset/"
DATA_NAME = 'D2-N'
DIRPATH = f"{ROOT_DIR}{DATA_NAME}"

df = _load.get_samples(DIRPATH, n_sequences=1000)
print(df.shape)
df.head()


# %%
# --------------------------------------------------
""" filter cdr3 region. NOTE: this is done by slicing the sequence
from the start of the v region `n` nucleotides based on the reported cdr3
length column """

def _apply_crd3_slice(row):
    """An apply function to extract the cdr3 region from the 
    sequence""" 
    seq = row['nucleotide']
    start = row['vIndex']
    end = start + row['cdr3Length']
    cdr3 = seq[start:end]
    return cdr3

# define the column in the sampled dataframe 
df['cdr3_sequence'] = df.apply(lambda row: _apply_crd3_slice(row), axis=1)

# %% 

df.shape

# %%
# --------------------------------------------------
""" compute distance matrix using real minimal edit distance
from the python package leveshtien. 
"""

reload(_dist)
A = _dist.matrix_levenshtien(df['cdr3_sequence'].tolist())
A.shape

# %%
# --------------------------------------------------
""" Try graph viz
"""
matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['figure.figsize'] = 18, 18
plt.style.use('seaborn-deep')

g = nx.from_numpy_matrix(A)

node_color = list(nx.information_centrality(g).values())
edge_weights = [g[u][v]['weight'] for u,v in g.edges()]

pos = nx.spring_layout(g, weight='weight')
# pos = nx.spectral_layout(g)
nx.draw_networkx(g, 
                 pos, 
                 with_labels = False, 
                 node_color = node_color, 
                 edgecolors = 'black',
                 width = edge_weights,
                 edge_cmap = plt.get_cmap('Greys_r'),
                 edge_color = edge_weights,
                 cmap = 'Spectral', 
                 alpha = 0.7) 


plt.suptitle(f"{DATA_NAME} Sample Network")

# %%
