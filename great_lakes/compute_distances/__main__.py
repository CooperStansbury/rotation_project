"""
Author:
    cstansbu

Description:
    main executable entry point for computing pairwise distances
"""
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

# local imports
import distance
import file_loader

# global variables 
SEQUENCE_COLUMN = 'nucleotide'
ROOT_DIR = f'/scratch/indikar_root/indikar1/shared_data/cstansbu_rotation/b_cell_data/'
OUTPUT_DIRECTORY = '/scratch/indikar_root/indikar1/shared_data/cstansbu_rotation/b_cell_data/outputs'
SAMPLE_SIZE = 10000

if __name__ == '__main__':
    print(f"RUNTIME: {datetime.today().strftime('%Y-%m-%d-%H:%M:%S')}")
    print(f"SEQUENCE_COLUMN: {SEQUENCE_COLUMN}")
    print(f"OUTPUT_DIRECTORY: {OUTPUT_DIRECTORY}")
    print(f"SAMPLE_SIZE: {SAMPLE_SIZE}")
    print()

    # all donor datasets
    _names = ['D1-Na', 'D1-Nb', 'D1-M', 
              'D2-N', 'D2-M',
              'D3-N', 'D3-M']

    for NAME in _names:
        INPUT_DIRECTORY = f"{ROOT_DIR}{NAME}"

        print(f"INPUT_DIRECTORY: {INPUT_DIRECTORY}")
        print(f"Data Name: {NAME}")

        # sample SAMPLE_SIZE files distributed as close
        # to evenly acrossed all files in ROOT_DIRECTORY
        n_files = len(os.listdir(INPUT_DIRECTORY))
        samples_per_file = int(SAMPLE_SIZE / n_files)

        sequence_list = []
        features_list = []

        for i, file in enumerate(os.listdir(INPUT_DIRECTORY)):
            if not file.endswith('.tsv'):
                continue # skip non-input tsv files

            file_path = f"{INPUT_DIRECTORY}/{file}" # path spec
            tmp_df = file_loader.read_file(file_path)

            # sample from each file, assuming there are enough records
            if samples_per_file < tmp_df.shape[0]:
                sample = tmp_df.sample(samples_per_file)
                sequence_list += sample[SEQUENCE_COLUMN].tolist()
                features_list.append(sample)

        # save the features for, say plotting
        df = pd.concat(features_list, ignore_index=True)
        output_file_path = f"{OUTPUT_DIRECTORY}/{NAME}_features.csv"
        df.to_csv(output_file_path, index=False)

        """
        This code block saves distance matrices
        """
        
        # build distance matrix
        distances = distance.matrix_levenshtien(sequence_list)
        print(f"distance frame shape: {distances.shape}")

        # save distances
        output_file_path = f"{OUTPUT_DIRECTORY}/{NAME}_distances.npy"
        np.save(output_file_path, distances)
        print(f"Done saving: {output_file_path}")
        print()




