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
import pandas as pd
from datetime import datetime
import scipy
import scipy.sparse
import scipy.linalg

# local imports
sys.path.insert(0, '/home/cstansbu/project_packages')

import umap 

# global variables 
INPUT_DIRECTORY = f'/scratch/indikar_root/indikar1/shared_data/cstansbu_rotation/b_cell_data/outputs'
INPUT_FILE = f'{INPUT_DIRECTORY}/D1-M_distances.npy'
INPUT_NAME = INPUT_FILE.split("/")[-1].replace('_distances.npy', '')
OUTPUT_DIRECTORY = '/scratch/indikar_root/indikar1/shared_data/cstansbu_rotation/b_cell_data/outputs'
K=100

print(f"RUNTIME: {datetime.today().strftime('%Y-%m-%d-%H:%M:%S')}")
print(f"UMAP Location: {umap.__file__}")
print(f"INPUT_FILE: {INPUT_FILE}")
print(f"INPUT_NAME: {INPUT_NAME}")
print(f"OUTPUT_DIRECTORY: {OUTPUT_DIRECTORY}")
print(f"K: {K}")
print()


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

# def get_knee(eigenvalues, S=2):
#     """Return the elbow of an order array of eigenvalues
    
#     Args:
#         - eigenvalues (np.array)
#         - S (int): sensitivity
        
#     Returns:
#         - knee_index (int): index at which 
#             the second derivititive changes 
#             most drastically + 1
#     """
#     ind = np.arange(len(eigenvalues)) + 1
#     knee = kneed.KneeLocator(ind, eigenvalues, S=S, 
#                           curve='convex', 
#                           direction='decreasing')
#     return knee.knee + 1


if __name__ == '__main__':

    A = np.load(INPUT_FILE)
    print(f"{INPUT_NAME} shape: {A.shape}")

    # compute the Laplacian
    L = scipy.sparse.csgraph.laplacian(A, normed=True) 

    # compute SVD of the Laplacian 
    w, v = np.linalg.eig(L)
    
    # normalize the first K eigenvectors 
    v = normlize_rows(v[:,0:K])

    # perform UMAP
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(v)

    print(embedding.shape)

    # save embeddings
    output_file_path = f"{OUTPUT_DIRECTORY}/{INPUT_NAME}_embeddings.npy"
    np.save(output_file_path, embedding)
    print(f"Done saving: {output_file_path}")











    

