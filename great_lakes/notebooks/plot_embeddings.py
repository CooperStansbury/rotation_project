

# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import os

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
                'D2-N', 'D2-M',
                'D3-N', 'D3-M']:


    if dataset == 'D2-M':
        df = utils.load_embed_and_feat(dataset, ROOT_DIR)
        print(dataset, df.shape)

        fig = utils.build_fig(y_lab="UMAP-2", 
                            x_lab="UMAP-1", 
                            title=f'Sample from {dataset} Clusters')

        sns.scatterplot(x=df['UMAP_1'], 
                        y=df['UMAP_2'], 
                        hue=df['vFamilyName'],
                        alpha=0.5)

    outpath = f"../figures/{dataset}_UMAP_by_Family.png"
    plt.savefig(outpath, bbox_inches = 'tight')

    

# %%
