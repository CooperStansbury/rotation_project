

# %%
# ---------------------------------------------------------------------------------------------
# dependencies 
import numpy as np
import pandas as pd
import os
import sys

# make local utils discoverable
sys.path.append('../')
from similarity_matrix_builder import file_loader


# %%
# ---------------------------------------------------------------------------------------------
# load the file
FILE_PATH = '/scratch/indikar_root/indikar1/shared_data/cstansbu_rotation/b_cell_data/test_data/D1-M_0_BRR_D1-M-188.adap.txt.results.tsv'

df = file_loader.read_file(FILE_PATH)
df.head()


# %%

[x for x in df.columns]
# %%

print(df['vIndex'].min())
print(df['vIndex'].max())
# %%

