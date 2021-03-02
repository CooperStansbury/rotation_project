

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


# %%
%matplotlib inline

# %%

def load_embed_and_feat(data_name, root_dir):
    """A function to load embeddings and features for a donor/cell type

    Args:
        - data_name (str): one of: 
            ['D1-Na', 'D1-Nb', 'D1-M', 
             'D2-N', 'D2-M',
             'D3-N', 'D3-M']

        - root_dir (str): path to the folder with all analysis files
    
    Returns:
        - df (pd.DataFrame): a data frame with embeddings as columns
    """

    for file in os.listdir(root_dir):
        if data_name in file: 
            # load features
            feat_path = f"{root_dir}{data_name}_features.csv"
            df = pd.read_csv(feat_path)

            # load embeddings and add to dataframe
            embed_path = f"{root_dir}{data_name}_embeddings.npy"
            dings = np.load(embed_path)
            ding_df = pd.DataFrame(dings, columns = ['UMAP_1','UMAP_2'])
            del dings

            df['UMAP_1'] = ding_df['UMAP_1']
            df['UMAP_2'] = ding_df['UMAP_2']

            del ding_df
    return df


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

ROOT_DIR = "/scratch/indikar_root/indikar1/shared_data/cstansbu_rotation/b_cell_data/outputs/"

for dataset in ['D1-Na', 'D1-Nb', 'D1-M',
                'D2-N', 'D1-M',
                'D3-N', 'D3-M']:

    df = load_embed_and_feat(dataset, ROOT_DIR)
    print(dataset, df.shape)

    fig = build_fig(y_lab="UMAP-2", 
                    x_lab="UMAP-1", 
                    title=f'Sample from {dataset} Clusters')

    sns.scatterplot(x=df['UMAP_1'], 
                    y=df['UMAP_2'], 
                    hue=df['vFamilyName'],
                    alpha=0.5)

    outpath = f"../figures/{dataset}_UMAP_by_Family.png"
    plt.savefig(outpath, bbox_inches = 'tight')

# %%
