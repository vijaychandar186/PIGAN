import random
from typing import Optional


class PhyloNode:
    """Node in a phylogenetic tree carrying a lineage label."""

    def __init__(self, value: str, lineage: str = ""):
        self.value = value
        self.lineage = lineage
        self.left: Optional["PhyloNode"] = None
        self.right: Optional["PhyloNode"] = None
        self.children: list = []


def pre_order_traversal(node: Optional[PhyloNode]) -> list:
    """
    Algorithm 1 from paper: pre-order tree traversal.
    Visits each node before any of its children (root → left → right).
    Ancestral ties are preserved: parent always precedes descendants.
    """
    if node is None:
        return []
    result = [node.value]
    for child in (node.children if node.children else [node.left, node.right]):
        result += pre_order_traversal(child)
    return result


def dfs_paths(node: Optional[PhyloNode]) -> list:
    """Return all root-to-leaf paths via DFS."""
    if node is None:
        return []
    children = [c for c in (node.children if node.children else [node.left, node.right]) if c is not None]
    if not children:
        return [[node]]
    paths = []
    for child in children:
        for path in dfs_paths(child):
            paths.append([node] + path)
    return paths


def pt_based_sampling(
    tree: PhyloNode,
    dataset: dict,
    samples_per_path: int,
    min_path_len: int = 5,
    max_path_len: int = 8,
) -> list:
    """
    Algorithm 2 from paper: PT-based sampling.
    Selects sequences from the dataset following evolutionary paths in the
    phylogenetic tree, filtered to paths of length [min_path_len, max_path_len].

    Args:
        tree: root PhyloNode of constructed phylogenetic tree
        dataset: dict mapping lineage → list of sequences
        samples_per_path: N, number of historical sequences sampled per path
        min_path_len: minimum path length (paper uses 5)
        max_path_len: maximum path length (paper uses 8)

    Returns:
        D: list of sampled historical sequence lists
    """
    all_paths = dfs_paths(tree)
    filtered = [p for p in all_paths if min_path_len <= len(p) <= max_path_len]

    training_dataset = []
    for path in filtered:
        for _ in range(samples_per_path):
            sample = []
            for node in path:
                candidates = dataset.get(node.lineage, [])
                if candidates:
                    sample.append(random.choice(candidates))
            if sample:
                training_dataset.append(sample)
    return training_dataset


def build_tree_from_newick(newick: str) -> PhyloNode:
    """
    Parse a Newick-format string into a PhyloNode tree.
    Requires biopython: `pip install biopython`.
    """
    from Bio import Phylo
    from io import StringIO
    bio_tree = Phylo.read(StringIO(newick), "newick")

    def _convert(clade) -> PhyloNode:
        name = clade.name or clade.branch_length or ""
        node = PhyloNode(value=str(name), lineage=str(name))
        node.children = [_convert(c) for c in clade.clades]
        return node

    return _convert(bio_tree.root)
