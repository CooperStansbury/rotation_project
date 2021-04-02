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
                            sep=sep,
                            low_memory=False)
    else:
        return  pd.read_csv(file_path, 
                    skiprows=n_header,
                    sep=sep,
                    low_memory=False)


def get_samples(dirpath, n_sequences=1000, sample_from=0.1):
    """A function to return a single dataframe
    of randomly sampled sequences.
    
    NOTE: this function randomly sample a certain
    number of files, then selects sequences from those files
    """

    SAMPLE_FROM = int(len(os.listdir(dirpath)) * sample_from)
    file_list = np.random.choice(os.listdir(dirpath), SAMPLE_FROM, replace=False)

    n_files = len(file_list)

    # oversample and trim
    samples_per_file = int((n_sequences / n_files) * 1.1)

    df_list = []

    for i, file in enumerate(file_list):
        if not file.endswith('.tsv'):
            continue # skip non-input tsv files

        file_path = f"{dirpath}/{file}"
        tmp_df = read_file(file_path)

        if samples_per_file < tmp_df.shape[0]:
            sample = tmp_df.sample(samples_per_file)
            df_list.append(sample)


    df = pd.concat(df_list, ignore_index=True)
    to_drop = -1* (len(df) - n_sequences)
    df = df[:to_drop]
    return df
        