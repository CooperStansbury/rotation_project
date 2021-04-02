"""
Author:
    cstansbu

Description:
    functions for building a similarity matrix
"""

# make levenstien package findable
import sys
sys.path.insert(0, '/home/cstansbu/project_packages')
import Levenshtein # note that this is local 

import pandas as pd
import numpy as np
from itertools import combinations


def pairwise_levenshtien(sequences):
    """A function to build pairwise distances using the 
    levenshtien distance. Only builds pairwise combinations.
    
    Args:
        - sequences (iterable): input will likley be a pandas column, numpy
        array, or list containing sequences.

    Returns:
        - distances (pd.Dataframe)
    """
    new_rows = []

    for i, pair in enumerate(combinations(sequences, 2)):
        row = {
            'seq1':pair[0],
            'seq2':pair[1],
            'distance': Levenshtein.distance(pair[0], pair[1])
        }

        new_rows.append(row)
    return pd.DataFrame(new_rows)



def matrix_levenshtien(sequences):
    """A function to build pairwise distance matrix using the 
    levenshtien distance. NOTE: this is very slow and should 
    be use with caution. 
    
    Args:
        - sequences (iterable): input will likley be a pandas column, numpy
        array, or list containing sequences.

    Returns:
        - distance_matrix (np.array)
    """
    N = len(sequences)
    distance_matrix = np.zeros((N,N),dtype=np.int)
    for i in range(0, N):
        for j in range(0, N):
            distance_matrix[i,j] = Levenshtein.distance(sequences[i],sequences[j])

    return distance_matrix