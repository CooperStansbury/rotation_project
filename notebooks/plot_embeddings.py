

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
%matplotlib inline


# %%

ROOT_DIR = "/scratch/indikar_root/indikar1/shared_data/cstansbu_rotation/b_cell_data/outputs/"

embeddings = {}

for file in os.listdir(ROOT_DIR):
    if 'embed' in file:
        data_name = file.split("_")[0]
        embed_fpath = f"{ROOT_DIR}{file}"
        dings = np.load(embed_fpath)
        df = pd.DataFrame(dings, columns = ['UMAP_1','UMAP_2'])
        embeddings[data_name] = df

print(embeddings.keys())

# %%


def build_fig(title="", axis_off=False, size=(10, 8), dpi=200, 
              y_lab="", x_lab=""):
    """A function to build a matplotlib figure. Primary
    goal is to sandardize the easy stuff.
    Args:
        - title (str): the title of the plot
        - axis_off (bool): should the axis be printed?
        - size (tuple): how big should the plot be?
        - y_lab (str): y axis label
        - x_lab (str): x axis label
    Returns:
        fig (plt.figure)
    """
    fig = plt.figure(figsize=size, 
                     facecolor='w',
                     dpi=dpi)
    fig.suptitle(title, fontsize=15)
    plt.xlabel(x_lab, fontsize=15)
    plt.ylabel(y_lab, fontsize=15)
    
    if axis_off:
        plt.axis('off')
    return fig


# %%

KEY = 'D3-M'

fig = build_fig(y_lab="UMAP-2", x_lab="UMAP-1", title=f'Sample from {KEY} Clusters')

sns.scatterplot(x=embeddings[KEY]['UMAP_1'], 
                y=embeddings[KEY]['UMAP_2'], 
                palette='Set1',
                alpha=0.5)

# %%
