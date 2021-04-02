"""
Author:
    cstansbu

Description:
    main executable entry point network visualization
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import os
import sys
import networkx as nx
from datetime import datetime

plt.switch_backend('agg')
matplotlib.rcParams['figure.dpi'] = 300
matplotlib.rcParams['figure.figsize'] = (20, 20)
plt.style.use('seaborn-deep')
from matplotlib import rcParams

# local imports
import utils

# global variables 
ROOT_DIR = "/scratch/indikar_root/indikar1/shared_data/cstansbu_rotation/b_cell_data/outputs/"
OUTPUT_DIRECTORY = "/home/cstansbu/rotation_project/figures/"

"/home/cstansbu/rotation_project/figures"


if __name__ == '__main__':
    print(f"RUNTIME: {datetime.today().strftime('%Y-%m-%d-%H:%M:%S')}")
    print(f"OUTPUT_DIRECTORY: {OUTPUT_DIRECTORY}")
    print()

    for dataset in ['D1-Na', 'D1-Nb', 'D1-M',
                'D2-N', 'D1-M',
                'D3-N', 'D3-M']:

        A = utils.load_adjacency(dataset, ROOT_DIR)
        print(dataset, A.shape)

        # convert to networkx object
        # g = nx.convert_matrix.from_numpy_matrix(A)
        g = nx.to_networkx_graph(A)
        print(f"done creating graph: {datetime.today().strftime('%Y-%m-%d-%H:%M:%S')}")

        # # compute harmonic centrality
        # node_color = list(nx.harmonic_centrality(g).values())

        # pos = nx.spring_layout(g)
        # nx.draw_networkx(g, 
        #                  pos, 
        #                  edgecolor = 'gray',
        #                  with_labels = False, 
        #                  alpha = 0.3,
        #                  node_size = 2,
        #                  cmap='Spectral_r') 
                            
        # outpath = f"{OUTPUT_DIRECTORY}{dataset}_network.png"
        # plt.savefig(outpath, bbox_inches = 'tight')

        break