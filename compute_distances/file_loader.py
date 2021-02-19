"""
Author:
    cstansbu

Description:
    File IO utilities
"""

import numpy as np
import pandas as pd
import sys
import os



def header_sniffer(file_path, char="#"):
        """A function to count the number of metadata 
        lines at the begining of a file. 
        
        Args:
            - file_path (str): full path to the file
            - char (str): the character used to signal header lines

        Returns:
            - n_header (int): the number of lines to skip
        """
        return np.sum([1 for line in open(file_path) if line.startswith(char)])


def read_file(file_path, usecols=None, sep='\t'):
    """A function to read a single file, adjusting for header records.

    Args:
        - file_path (str): full path to the file
        - usecols (bool): if list of colnames supplied, only those
        column names will be returned.
        - sep (str): deliminter

    Returns:
        - df (pd.DataFrame): a dataframe of values
    """
    n_header = header_sniffer(file_path)

    if usecols is not None:
        return  pd.read_csv(file_path, 
                            skiprows=n_header,
                            usecols=usecols,
                            sep=sep)
    else:
        return  pd.read_csv(file_path, 
                    skiprows=n_header,
                    sep=sep)