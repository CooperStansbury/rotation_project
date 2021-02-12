"""
Author:
    cstansbu

Description:
    functions for building a similarity matrix
"""

import pandas as pd
import numpy as np
import Levenshtein # note that this is local 
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from itertools import combinations
from time import gmtime, strftime

def flatten_sequences(sequences):
    """Get all pairwise combinations from list of sequences

    Args:
        - sequences (iterable): input will likley be a pandas column, numpy
            array, or list containing sequences.
    
    Returns:
        - pairs (fruit, jk tuples): all pairs of sequences
    """
    return np.array(combinations(sequences, 2))


def slow_levenshtien_distance(sequences):
    """A function to build pairwise distance matrix
    using the levenshtien distance.

    Args:
        - sequences (iterable): input will likley be a pandas column, numpy
        array, or list containing sequences.

    Returns:
        - distance_matrix (np.array)
    """
    N = len(sequences)
    distance_matrix = np.zeros((N,N),dtype=np.int)
    for i in range(0, N):
        
        # stoopid time logging
        if i % 1000 == 0:
            curr_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
            print(f"row {i}: {curr_time}")

        for j in range(0, N):
            distance_matrix[i,j] = Levenshtein.distance(sequences[i],sequences[j])

    return distance_matrix

