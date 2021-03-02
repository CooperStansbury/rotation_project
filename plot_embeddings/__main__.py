"""
Author:
    cstansbu

Description:
    main executable entry point for plotting embeddings
"""
import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime

# general plotting config
plt.switch_backend('agg')
matplotlib.rcParams['figure.dpi'] = 300
plt.style.use('seaborn-deep')
from matplotlib import rcParams

# local imports
import plotting_utils

# global variables 
ROOT_DIR = "/scratch/indikar_root/indikar1/shared_data/cstansbu_rotation/b_cell_data/outputs/"
DATA_NAME = 'D1-M'
OUTPUT_DIRECTORY = '/home/cstansbu/rotation_project/figures/'

"""
FUNCTIONS
"""

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


if __name__ == '__main__':
    # printing for the logs
    print(f"RUNTIME: {datetime.today().strftime('%Y-%m-%d-%H:%M:%S')}")
    print(f"DATA_NAME: {DATA_NAME}")
    print(f"OUTPUT_DIRECTORY: {OUTPUT_DIRECTORY}")
    print()


    df = load_embed_and_feat(DATA_NAME, ROOT_DIR)
    print(df.shape)
 
