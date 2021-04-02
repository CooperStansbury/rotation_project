# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import os
import networkx as nx

plt.switch_backend('agg')
matplotlib.rcParams['figure.dpi'] = 300
plt.style.use('seaborn-deep')
from matplotlib import rcParams

# load utiliies
import utils

# %%
%matplotlib inline

# %%

ROOT_DIR = "/scratch/indikar_root/indikar1/shared_data/cstansbu_rotation/b_cell_data/outputs/"

for dataset in ['D1-Na', 'D1-Nb', 'D1-M',
                'D2-N', 'D1-M',
                'D3-N', 'D3-M']:

    A = utils.load_adjacency(dataset, ROOT_DIR)
    print(dataset, A.shape)

    # convert to networkx object
    g = nx.convert_matrix.from_numpy_matrix(A)

    pos = nx.spring_layout(g)
    nx.draw_networkx(g, 
                     pos, 
                     with_labels = False, 
                     edgecolors = 'black',
                     alpha = 0.8,
                     node_size = 1) 
    plt.show()
    # outpath = f"figs/pol_books3a.png"
    # plt.savefig(outpath, bbox_inches = 'tight')

    break


# %%
