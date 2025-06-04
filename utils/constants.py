SUBTYPE_FLAGS = {'H1N1': 0, 'H3N2': 1, 'H5N1': 2, 'COV19': 3}
AMINO_ACIDS = ['A', 'F', 'Q', 'R', 'T', 'Y', 'V', 'I', 'H', 'K', 'P', 'N', 'E', 'G', 'S', 'M', 'D', 'W', 'C', 'L', '-', 'B', 'J', 'Z', 'X']
PROT_VEC_PATH = './processed/COV19/protVec_100d_3grams.csv'

def select_subtype(subtype: str) -> int:
    """Select subtype and return its flag."""
    return SUBTYPE_FLAGS.get(subtype, 3)  # Default to COV19