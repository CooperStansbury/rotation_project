"""
Author:
    cstansbu

Description:
    main executable entry point for umap clustering on seqeunce distances
"""

import os
import sys
from types import prepare_class
import numpy as np
from numpy.core.fromnumeric import var
import pandas as pd
from datetime import datetime
import scipy
import scipy.sparse
import scipy.linalg
from sklearn.cluster import SpectralClustering

# plotting utilities
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# plotting config
plt.switch_backend('agg')
matplotlib.rcParams['figure.dpi'] = 300
plt.style.use('seaborn-deep')
from matplotlib import rcParams

# local imports
sys.path.insert(0, '/home/cstansbu/project_packages')
import umap 

# global variables 
INPUT_DIRECTORY = f'/scratch/indikar_root/indikar1/shared_data/cstansbu_rotation/b_cell_data/outputs'
OUTPUT_DIRECTORY = f'/scratch/indikar_root/indikar1/shared_data/cstansbu_rotation/b_cell_data/outputs'

"""
FUNCTIONS
"""

def normlize_rows(eigenvectors):
    """A fucntion to normalize eigenvectors
    row-wise across n eigenvectors
    
    Args:
        - eigenvectors (np.array): the n eigenvectors
    
    Returns:
        - T (np.array): eigenvectors normalized
    """
    T = np.zeros(eigenvectors.shape)
    for idx, row in enumerate(eigenvectors):
        T[idx] = abs(row / np.linalg.norm(row, ord=1))
    return T


def build_fig(title="", axis_off=False, size=(10, 8), 
              y_lab="", x_lab="", title_size=15):
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
                     dpi=300)
    fig.suptitle(title, fontsize=title_size)
    plt.xlabel(x_lab, fontsize=15)
    plt.ylabel(y_lab, fontsize=15)
    
    if axis_off:
        plt.axis('off')
    return fig


def plot_knee(eigs, K,  input_name, outpath, log=True):
    """A function to plot the knee of the eigendistribution.

    Args:
        - eigs (np.array): the eigenvalues of the matrix
        - K (int): the index value of the inflection point
        - input_name (str): the name of the dataset for the figure title.
        - outpath (str): location to save the figure
        - log (bool): if true, eigenvalues with be converted to log scale

    Returns:
        - None. Plotting only, prints status to log.
    """
    y_lab = 'Singular Value'

    if log:
        y_lab = 'Singular Value (log)'
        _eigs = eigs + np.finfo(float).eps
        eigs = np.log(eigs)

    idx = sing_vals = np.arange(len(eigs)) + 1
    fig = build_fig(title=f'Singular Values for {input_name}',
                    x_lab='Single Value Index',
                    y_lab=y_lab)
    
    # plot singular values
    plt.plot(idx, eigs, 
            color='blue',    
            marker='.')

    # add knee marker
    plt.axvline(x=K, color='red', linestyle='--')

    plt.savefig(outpath)
    print(f"done saving knee plot to: {outpath}")


if __name__ == '__main__':
    print(f"RUNTIME: {datetime.today().strftime('%Y-%m-%d-%H:%M:%S')}")
    print(f"OUTPUT_DIRECTORY: {OUTPUT_DIRECTORY}")
    print()


    # the different datasets
    _names = ['D1-Na', 'D1-Nb', 'D1-M', 
              'D2-N', 'D2-M',
              'D3-N', 'D3-M']

    # process all data in single run
    for NAME in _names:
        INPUT_FILE = f'{INPUT_DIRECTORY}/{NAME}_distances.npy'
        print(f"INPUT_FILE: {INPUT_FILE}")
        print(f"Data NAME: {NAME}")

        # load distance matrix (adjacency)
        A = np.load(INPUT_FILE)
        print(f"{NAME} shape: {A.shape}")

        # fill diagonal 
        np.fill_diagonal(A, 1)

        # normalize the matrix 
        A = A - A.mean()

        # compute SVD of the adjacency matrix 
        l_evecs, evals, _ = np.linalg.svd(A)

        # compute r 
        coeff = np.sqrt(4) / 3
        r = coeff * np.median(evals)

        s_ind = np.argwhere(evals >= r)
        k = np.max(s_ind)

        print(f"estimated hard threshold: {r} occurs at index {k}")

        # save the knee plot:
        outpath = f"/home/cstansbu/rotation_project/figures/{NAME}_knee.png"
        plot_knee(evals, k, NAME, outpath)

        # compute v
        P = l_evecs[:,0:k]
        clustering = SpectralClustering(n_clusters=k, random_state=1729).fit(P)

        # perform UMAP
        reducer = umap.UMAP()
        embedding = reducer.fit_transform(P)
        print(embedding.shape)

        # save embeddings
        output_file_path = f"{OUTPUT_DIRECTORY}/{NAME}_embeddings.npy"
        np.save(output_file_path, embedding)
        print(f"Done saving: {output_file_path}")
        print()

        # plot clusters 
        fig = build_fig(y_lab="UMAP-2", 
                        x_lab="UMAP-1", 
                        title=f'Sample from {NAME} Clusters')

        sns.scatterplot(x=embedding[:, 0], 
                        y=embedding[:, 1], 
                        hue=clustering.labels_,
                        alpha=0.5)

        outpath = f"/home/cstansbu/rotation_project/figures/{NAME}_UMAP_by_Cluster_Label.png"
        plt.savefig(outpath, bbox_inches = 'tight')

        # load the metadata for the file 
        feat_path = f"{INPUT_DIRECTORY}/{NAME}_features.csv"
        df = pd.read_csv(feat_path)

        # plot the clusters by vfamily name
        # plot clusters 
        fig = build_fig(y_lab="UMAP-2", 
                        x_lab="UMAP-1", 
                        title=f'Sample from {NAME}, by Family Name')

        sns.scatterplot(x=embedding[:, 0], 
                        y=embedding[:, 1], 
                        hue=df['vFamilyName'],
                        alpha=0.5)

        outpath = f"/home/cstansbu/rotation_project/figures/{NAME}_UMAP_by_Family_Name.png"
        plt.savefig(outpath, bbox_inches = 'tight')










    

