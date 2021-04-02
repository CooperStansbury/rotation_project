

"""
To count the total number of seqs per donor and cell type
"""

# %%

import sys
import os
import pandas

sys.path.append('../')
from get_distance_pairs import file_loader


# %%
ROOT_DIR = '/scratch/indikar_root/indikar1/shared_data/cstansbu_rotation/b_cell_data/'

for _dir in os.listdir(ROOT_DIR):
    subdir_path = f"{ROOT_DIR}{_dir}"
    if os.path.isdir(subdir_path):
        print(f"{_dir} has {len(os.listdir(subdir_path))} files ")
    else:
        continue
"""
Output:
>>> D1-Na has 188 files 
>>> outputs has 0 files 
>>> D2-M has 188 files 
>>> D2-N has 188 files 
>>> test_data has 2 files 
>>> D1-M has 188 files 
>>> D3-N has 188 files 
>>> D1-Nb has 188 files 
>>> D3-M has 188 files 
"""

# %%

for _dir in os.listdir(ROOT_DIR):
    subdir_path = f"{ROOT_DIR}{_dir}"

    subdir_count = 0
    if os.path.isdir(subdir_path):
        for file in os.listdir(subdir_path):
            if not file.endswith('.tsv'):
                continue 

            file_path = f"{subdir_path}/{file}"
            df = file_loader.read_file(file_path, usecols=['nucleotide'])
            subdir_count += df.shape[0]

    print(f"{_dir} has {subdir_count} sequences")


"""
Output:
>>> D1-Na has 7995966 sequences
>>> outputs has 0 sequences
>>> D2-M has 8418156 sequences
>>> D2-N has 6006650 sequences
>>> D1-M_0_BRR_D1-M-001.adap.txt.results.tsv has 0 sequences
>>> test_data has 39656 sequences
>>> D1-M has 8223221 sequences
>>> D3-N has 8431449 sequences
>>> D1-Nb has 7400302 sequences
>>> D3-M has 9785485 sequences
"""


# %%
