import random
import numpy as np
from utils.constants import AMINO_ACIDS

class Trigram:
    """Represents a trigram (three amino acids) with its starting position in a strain sequence."""
    def __init__(self, amino_acids: str, strain_pos: int):
        self.amino_acids = amino_acids
        self.strain_pos = strain_pos

    def contains_position(self, pos: int) -> bool:
        """Check if a given position is within the trigram's range."""
        return self.strain_pos <= pos < self.strain_pos + len(self.amino_acids)

def sample_strains_by_year(strains_by_year: list, num_samples: int) -> list:
    """Randomly sample strains from each year."""
    return [random.choices(year_strains, k=num_samples) for year_strains in strains_by_year]

def sample_strains_by_cluster(strains_by_year: list, num_samples: int) -> list:
    """Sample strains by selecting the first num_samples from each year."""
    return [year_strains[:num_samples] for year_strains in strains_by_year]

def split_into_trigrams(strains_by_year: list, overlapping: bool = True) -> list:
    """Split strains into trigrams, optionally overlapping."""
    step_size = 1 if overlapping else 3
    num_trigrams = len(strains_by_year[0][0]) - 2 if overlapping else len(strains_by_year[0][0]) // step_size
    trigrams_by_year = []

    for year_strains in strains_by_year:
        year_trigrams = []
        for strain in year_strains:
            strain_trigrams = [
                Trigram(strain[i * step_size:i * step_size + 3], i * step_size)
                for i in range(num_trigrams)
            ]
            remainder = len(strain) % step_size
            if remainder > 0:
                padding = '-' * (3 - remainder)
                strain_trigrams.append(Trigram(strain[-remainder:] + padding, len(strain) - remainder))
            year_trigrams.append(strain_trigrams)
        trigrams_by_year.append(year_trigrams)
    return trigrams_by_year

def create_triplet_strains(strains_by_year: list, positions: list) -> list:
    """Create triplet strains centered around specified positions with padding."""
    triplet_strains_by_year = []
    triplet_margin = 2
    for year_strains in strains_by_year:
        triplet_strains = []
        for strain in year_strains:
            for pos in positions:
                if pos < triplet_margin:
                    padding_size = triplet_margin - pos
                    triplet = '-' * padding_size + strain[:pos + triplet_margin + 1]
                elif pos > len(strain) - 1 - triplet_margin:
                    padding_size = pos - (len(strain) - 1 - triplet_margin)
                    triplet = strain[pos - triplet_margin:] + '-' * padding_size
                else:
                    triplet = strain[pos - triplet_margin:pos + triplet_margin + 1]
                triplet_strains.append(triplet)
        triplet_strains_by_year.append(triplet_strains)
    return triplet_strains_by_year

def create_triplet_labels(triplet_strains_by_year: list) -> list:
    """Create binary labels for triplet strains (0 if same, 1 if different)."""
    num_triplets = len(triplet_strains_by_year[0])
    epitope_pos = 2
    return [
        0 if triplet_strains_by_year[-1][i][epitope_pos] == triplet_strains_by_year[-2][i][epitope_pos] else 1
        for i in range(num_triplets)
    ]

def compute_majority_baselines(triplet_strains_by_year: list, labels: list) -> tuple:
    """Compute baseline performance using majority voting."""
    epitope_pos = 2
    predictions = []
    for i in range(len(labels)):
        epitopes = [year_strains[i][epitope_pos] for year_strains in triplet_strains_by_year[:-1]]
        majority_epitope = max(set(epitopes), key=epitopes.count)
        predictions.append(0 if triplet_strains_by_year[-2][i][epitope_pos] == majority_epitope else 1)
    conf_matrix = compute_confusion_matrix(np.array(labels), np.array(predictions))
    return (
        compute_accuracy(conf_matrix),
        compute_precision(conf_matrix),
        compute_recall(conf_matrix),
        compute_f1_score(conf_matrix),
        compute_mcc(conf_matrix)
    )

def extract_trigrams_by_positions(positions: list, trigrams_by_year: list) -> list:
    """Extract trigrams containing specified positions."""
    strain = trigrams_by_year[0][0]
    indices_to_extract = []
    idx = 0
    for pos in positions:
        pos_found = False
        while not pos_found and idx < len(strain):
            if strain[idx].contains_position(pos):
                pos_found = True
            else:
                idx += 1
        while idx < len(strain) and strain[idx].contains_position(pos):
            indices_to_extract.append(idx)
            idx += 1
    return [[strain_trigrams[i] for i in indices_to_extract] for year_trigrams in trigrams_by_year for strain_trigrams in year_trigrams]

def flatten_trigrams(trigrams_by_year: list) -> list:
    """Flatten trigrams by year into a single list per year."""
    return [[trigram for strain_trigrams in year_trigrams for trigram in strain_trigrams] for year_trigrams in trigrams_by_year]

def replace_uncertain_amino_acids(amino_acids: str) -> str:
    """Replace uncertain amino acids with a random valid amino acid."""
    replacements = {'B': 'DN', 'J': 'IL', 'Z': 'EQ', 'X': 'ACDEFGHIKLMNPQRSTVWY'}
    for uncertain, options in replacements.items():
        amino_acids = amino_acids.replace(uncertain, random.choice(options))
    return amino_acids

def map_trigrams_to_indices(nested_trigram_list: list, trigram_to_idx: dict) -> list:
    """Map trigrams to their indices, handling padding with a dummy index."""
    dummy_idx = len(trigram_to_idx)
    def mapping(trigram):
        if isinstance(trigram, Trigram):
            amino_acids = replace_uncertain_amino_acids(trigram.amino_acids)
            return trigram_to_idx.get(amino_acids, dummy_idx) if '-' not in amino_acids else dummy_idx
        elif isinstance(trigram, list):
            return [mapping(item) for item in trigram]
        raise TypeError(f"Expected nested list of Trigrams, but encountered {type(trigram)}")
    return [mapping(item) for item in nested_trigram_list]

def map_indices_to_vectors(nested_idx_list: list, idx_to_vec: np.ndarray) -> list:
    """Map indices to their corresponding vectors, using a dummy vector for padding."""
    dummy_vec = idx_to_vec[-1]
    def mapping(idx):
        if isinstance(idx, int):
            return idx_to_vec[idx] if idx < idx_to_vec.shape[0] else dummy_vec
        elif isinstance(idx, list):
            return [mapping(i) for i in idx]
        raise TypeError(f"Expected nested list of ints, but encountered {type(idx)}")
    return [mapping(item) for item in nested_idx_list]

def compute_difference_vectors(trigram_vecs_by_year: np.ndarray) -> np.ndarray:
    """Compute difference vectors between consecutive years."""
    diff_vecs = np.zeros((trigram_vecs_by_year.shape[0] - 1, trigram_vecs_by_year.shape[1], trigram_vecs_by_year.shape[2]))
    for i in range(diff_vecs.shape[0]):
        diff_vecs[i] = trigram_vecs_by_year[i + 1] - trigram_vecs_by_year[i]
    return diff_vecs

def detect_mutations(trigram_indices_x: list, trigram_indices_y: list) -> np.ndarray:
    """Detect mutations by comparing trigram indices."""
    assert len(trigram_indices_x) == len(trigram_indices_y)
    return np.array([1 if x != y else 0 for x, y in zip(trigram_indices_x, trigram_indices_y)])

def reshape_to_linear_features(vecs_by_year: np.ndarray, window_size: int = 3) -> list:
    """Reshape vectors into a linear format for baseline models."""
    reshaped = [[] for _ in range(len(vecs_by_year[0]))]
    for year_vecs in vecs_by_year[-window_size:]:
        for i, vec in enumerate(year_vecs):
            reshaped[i].extend(vec.tolist())
    return reshaped