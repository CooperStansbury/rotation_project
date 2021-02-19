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
INPUT_DIRECTORY = f'/scratch/indikar_root/indikar1/shared_data/cstansbu_rotation/b_cell_data/D1-Na'
INPUT_NAME = INPUT_DIRECTORY.split("/")[-1]
OUTPUT_DIRECTORY = '/scratch/indikar_root/indikar1/shared_data/cstansbu_rotation/b_cell_data/outputs'
SAMPLE_SIZE = 50000

print(f"RUNTIME: {datetime.today().strftime('%Y-%m-%d-%H:%M:%S')}")
print(f"SEQUENCE_COLUMN: {SEQUENCE_COLUMN}")
print(f"INPUT_DIRECTORY: {INPUT_DIRECTORY}")
print(f"INPUT_NAME: {INPUT_NAME}")
print(f"OUTPUT_DIRECTORY: {OUTPUT_DIRECTORY}")
print(f"SAMPLE_SIZE: {SAMPLE_SIZE}")
print()


if __name__ == '__main__':
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
        df = file_loader.read_file(file_path)

        # sample from each file, assuming there are enough records
        if samples_per_file < df.shape[0]:
            sequences = df[SEQUENCE_COLUMN].sample(samples_per_file)
            sequence_list += sequences.tolist()

            features_list.append(df)

    # save the features for, say plotting
    df = pd.concat(features_list, ignore_index=True)
    output_file_path = f"{OUTPUT_DIRECTORY}/{INPUT_NAME}_features.csv"
    df.to_csv(output_file_path, index=False)


    """
    This code block saves pairwaise distances
    """
    # # build pairwise distances
    # distances = distance.pairwise_levenshtien(sequence_list)
    # 
    # # save distances
    # output_file_path = f"{OUTPUT_DIRECTORY}/{INPUT_NAME}_distances.csv"
    # distances.to_csv(output_file_path, index=False)
    # print(f"Done saving: {output_file_path}")

    """
    This code block saves distance matrices
    """
    # build distance matrix
    distances = distance.matrix_levenshtien(sequence_list)
    print(f"distance frame shape: {distances.shape}")

    # save distances
    output_file_path = f"{OUTPUT_DIRECTORY}/{INPUT_NAME}_distances.npy"
    np.save(output_file_path, distances)
    print(f"Done saving: {output_file_path}")




