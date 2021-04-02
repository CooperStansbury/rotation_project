"""
Author:
    cstansbu

Description:
    function related to distances and distance
    matrix creation
"""

import numpy as np
import pandas as pd
import scipy
from itertools import combinations
from itertools import permutations
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import Levenshtein 


def get_distance_matrix(sequence_list):
    """A function to get the a levenshtien ratio matrix a list of 
    sequences.

    WARNING: this is very slow

    Args:
        - sequence_list (list): list of sequences

    Returns:
        A (n x n np.array2D): symetric distance matrix
    """
    # get all pairs in m x n array    
    pairs = permutations(sequence_list, 2)
    pairs_arr = np.array([*pairs])

    # force pairwise string conversion
    levvy = lambda u, w: Levenshtein.ratio(str(u), str(w))
    A = pdist(pairs_arr, levvy)
    A = squareform(A)
    A = np.fill_diagonal(A, 1)
    return squareform(A)
    # return A


def matrix_levenshtien(sequence_list):
    """A function to build pairwise distance matrix using the 
    levenshtien distance. NOTE: this is very slow and should 
    be use with caution. 
    
    Args:
        - sequence_list (iterable): input will likley be a pandas column, numpy
        array, or list containing sequences.

    Returns:
        - A (np.array): distance matrix
    """
    N = len(sequence_list)
    A = np.zeros((N,N), dtype=np.float64)

    for i in range(0, N):
        for j in range(0, N):
            levy_dist = Levenshtein.ratio(sequence_list[i],sequence_list[j])
            A[i,j] = levy_dist

    np.fill_diagonal(A, 1)
    return A