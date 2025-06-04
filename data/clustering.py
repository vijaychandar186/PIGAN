from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import random
import numpy as np
from utils.constants import AMINO_ACIDS

def encode_strains(strains: list) -> list:
    """Encode amino acid sequences using LabelEncoder."""
    le = LabelEncoder()
    le.fit(AMINO_ACIDS)
    return [le.transform(list(strain)) for strain in strains]

def decode_strains(encoded_strains: list) -> list:
    """Decode encoded strains back to amino acid sequences."""
    le = LabelEncoder()
    le.fit(AMINO_ACIDS)
    return [[''.join(le.inverse_transform(encoded_strain)) for encoded_strain in year_strains] for year_strains in encoded_strains]

def cluster_strains(strains: list, num_clusters: int = 2) -> dict:
    """Cluster strains using KMeans."""
    encoded = encode_strains(strains)
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(encoded)
    return {'data': encoded, 'labels': kmeans.labels_, 'centroids': kmeans.cluster_centers_}

def visualize_clusters(cluster: dict, save_path: str = 'none') -> None:
    """Visualize clusters in 2D using PCA."""
    encoded_strains = cluster['data']
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(encoded_strains)
    plt.figure()
    colors = ['r.', 'g.', 'y.', 'c.', 'm.', 'b.', 'k.'] * 10
    for i, (x, y) in enumerate(reduced_data):
        plt.plot(x, y, colors[cluster['labels'][i]], markersize=10)
    plt.show()
    if save_path != 'none':
        plt.savefig(save_path)
        plt.close()

def link_clusters_across_years(clusters_by_year: list) -> None:
    """Link clusters across years using nearest neighbors."""
    num_years = len(clusters_by_year)
    neigh = NearestNeighbors(n_neighbors=2)
    for year_idx in range(num_years):
        if year_idx == num_years - 1:
            clusters_by_year[year_idx]['links'] = []
            break
        links = []
        current_centroids = clusters_by_year[year_idx]['centroids']
        next_centroids = clusters_by_year[year_idx + 1]['centroids']
        neigh.fit(next_centroids)
        neighbor_indices = neigh.kneighbors(current_centroids, return_distance=False)
        for label in clusters_by_year[year_idx]['labels']:
            links.append(neighbor_indices[label])
        clusters_by_year[year_idx]['links'] = links

def sample_from_clusters(clusters_by_year: list, sample_size: int) -> list:
    """Sample strains across years following cluster links."""
    sampled_strains = []
    for _ in range(sample_size):
        sample = []
        start_idx = random.randint(0, len(clusters_by_year[0]['data']) - 1)
        sample.append(clusters_by_year[0]['data'][start_idx])
        current_idx = start_idx
        for year_idx in range(len(clusters_by_year) - 1):
            next_label = clusters_by_year[year_idx]['links'][current_idx][0]
            candidate_indices = np.where(clusters_by_year[year_idx + 1]['labels'] == next_label)[0]
            current_idx = random.choice(candidate_indices)
            sample.append(clusters_by_year[year_idx + 1]['data'][current_idx])
        sampled_strains.append(sample)
    return sampled_strains

def create_dataset(strains: list, position: int, window_size: int = 10, output_path: str = 'none') -> pd.DataFrame:
    """Create a dataset of trigram indices and labels for a given position."""
    labels = [0 if sample[-1][position] == sample[-2][position] else 1 for sample in strains]
    df = pd.read_csv(PROT_VEC_PATH, sep='\t')
    trigram_to_idx = {row['words']: index for index, row in df.iterrows()}
    start_year_idx = len(strains[0]) - window_size - 1
    data = []

    for sample in strains:
        sample_data = []
        for year in range(start_year_idx, len(sample) - 1):
            tritri = []
            if position == 0:
                tritri.extend([9047, 9047, trigram_to_idx.get(sample[year][position:position + 3], 9047)])
            elif position == 1:
                tritri.extend([
                    9047,
                    trigram_to_idx.get(sample[year][position - 1:position + 2], 9047),
                    trigram_to_idx.get(sample[year][position:position + 3], 9047)
                ])
            elif position == 1271:
                tritri.extend([
                    trigram_to_idx.get(sample[year][position - 2:position + 1], 9047),
                    trigram_to_idx.get(sample[year][position - 1:position + 2], 9047),
                    9047
                ])
            elif position == 1272:
                tritri.extend([trigram_to_idx.get(sample[year][position - 2:position + 1], 9047), 9047, 9047])
            else:
                tritri.extend([
                    trigram_to_idx.get(sample[year][position - 2:position + 1], 9047),
                    trigram_to_idx.get(sample[year][position - 1:position + 2], 9047),
                    trigram_to_idx.get(sample[year][position:position + 3], 9047)
                ])
            sample_data.append(str(tritri))
        data.append(sample_data)

    dataset = pd.DataFrame(data)
    dataset.insert(0, 'y', labels)
    if output_path != 'none':
        dataset.to_csv(output_path, index=False)
    return dataset