"""
Author:
    cstansbu

Description:
    main executable for the simlarity matrix builder tools
"""

import os
import sys
import distance
import file_loader
import numpy as np

# global variables
SEQUENCE_COLUMN = 'nucleotide'
INPUT_DIRECTORY = '/scratch/indikar_root/indikar1/shared_data/cstansbu_rotation/b_cell_data/test_data'
OUTPUT_DIRECTORY = '/scratch/indikar_root/indikar1/shared_data/cstansbu_rotation/b_cell_data/test_data'


if __name__ == '__main__':
    # right now only testing a single file
    for file in os.listdir(INPUT_DIRECTORY):
        file_path = f"{INPUT_DIRECTORY}/{file}"
        df = file_loader.read_file(file_path, usecols=[SEQUENCE_COLUMN])
        sequences = df[SEQUENCE_COLUMN].tolist()
        dist_mat = distance.slow_levenshtien_distance(sequences)
        print(dist_mat.shape)

        output_file_path = f"{OUTPUT_DIRECTORY}/test.npy"
        np.save(output_file_path, dist_mat)
        print("done saving.")

        break
    

