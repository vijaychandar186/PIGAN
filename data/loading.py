import pandas as pd
import os
import numpy as np
import ast
from utils.constants import SUBTYPE_FLAGS, PROT_VEC_PATH
from data.processing import map_indices_to_vectors

def select_subtype(subtype: str) -> int:
    """Select subtype and return its flag."""
    return SUBTYPE_FLAGS.get(subtype, 3)  # Default to COV19

def load_trigram_vectors(subtype: str) -> tuple:
    """Load trigram vectors and their indices from ProtVec file."""
    df = pd.read_csv(PROT_VEC_PATH, delimiter='\t')
    trigram_to_idx = {trigram: i for i, trigram in enumerate(df['words'])}
    trigram_vectors = df.loc[:, df.columns != 'words'].values
    return trigram_to_idx, trigram_vectors

def load_strains(data_files: list, data_path: str) -> list:
    """Load strain sequences from CSV files."""
    return [pd.read_csv(os.path.join(data_path, file_name))['seq'] for file_name in data_files]

def split_train_test_strains(strains_by_year: list, test_split: float, cluster: str) -> tuple:
    """Split strains into training and testing sets."""
    train_strains, test_strains = [], []
    for strains in strains_by_year:
        num_train = int(math.floor(strains.count() * (1 - test_split)))
        if cluster == 'random':
            shuffled = strains.sample(frac=1).reset_index(drop=True)
            train = shuffled.iloc[:num_train].reset_index(drop=True)
            test = shuffled.iloc[num_train:].reset_index(drop=True)
        else:
            train = strains.iloc[:800].reset_index(drop=True)
            test = strains.iloc[800:1000].reset_index(drop=True)
        train_strains.append(train)
        test_strains.append(test)
    return train_strains, test_strains

def load_dataset(path: str, data_path: str, limit: int = 0, concat: bool = False) -> tuple:
    """Load dataset of trigram vectors and labels."""
    _, trigram_vectors = load_trigram_vectors('COV19')
    df = pd.read_csv(path)
    if limit > 0:
        df = df.head(limit)
    labels = df['y'].values
    trigram_idx_strings = df.loc[:, df.columns != 'y'].values
    parsed_indices = [[ast.literal_eval(x) for x in example] for example in trigram_idx_strings]
    trigram_vecs = np.array(map_indices_to_vectors(parsed_indices, trigram_vectors))
    if concat:
        trigram_vecs = np.reshape(trigram_vecs, [len(df.columns) - 1, len(df.index), -1])
    else:
        trigram_vecs = np.sum(trigram_vecs, axis=2)
        trigram_vecs = np.moveaxis(trigram_vecs, 1, 0)
    return trigram_vecs, labels