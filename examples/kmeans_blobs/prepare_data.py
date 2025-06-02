# coding: utf-8

import os
import numpy as np
from sklearn.datasets import make_blobs
from typing import Literal, Optional
import fire

from declearn.dataset.utils import save_data_array

DATADIR = os.path.join(os.path.dirname(__file__), "data")

def prepare_blobs(
    nb_clients: int,
    n_samples: int = 1000,
    n_features: int = 2,
    centers: int = 3,
    cluster_std: float = 1.0,
    scheme: Literal["iid", "clusters"] = "iid",
    folder: str = DATADIR,
    seed: Optional[int] = 42,
) -> str:
    """Generate and split synthetic blob data for federated learning."""
    # Generate global blob dataset
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=centers,
        cluster_std=cluster_std,
        random_state=seed
    )
    
    # Split data based on scheme
    print(f"Splitting blob data into {nb_clients} shards using '{scheme}' scheme.")
    if scheme == "iid":
        # Random IID split
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(X))
        shard_indices = np.array_split(indices, nb_clients)
    elif scheme == "clusters":
        # Cluster-based split (each client gets one cluster)
        shard_indices = [np.where(y == i)[0] for i in range(centers)]
        # Handle case where nb_clients != centers
        if nb_clients > centers:
            # Duplicate clusters for extra clients
            shard_indices += [shard_indices[i % centers] for i in range(centers, nb_clients)]
        elif nb_clients < centers:
            # Merge clusters
            shard_indices = [np.concatenate(shard_indices[i::nb_clients]) for i in range(nb_clients)]
    else:
        raise ValueError(f"Unknown scheme: {scheme}")
    
    # Create output folder structure
    folder = os.path.join(folder, f"blobs_{scheme}")
    os.makedirs(folder, exist_ok=True)
    
    # Save data for each client
    for idx, indices in enumerate(shard_indices):
        client_dir = os.path.join(folder, f"client_{idx}")
        os.makedirs(client_dir, exist_ok=True)
        
        # Save training data
        save_data_array(os.path.join(client_dir, "train_data"), X[indices])
        save_data_array(os.path.join(client_dir, "train_target"), np.zeros(len(indices)))  # Dummy labels
        
        # Create empty validation files
        save_data_array(os.path.join(client_dir, "valid_data"), np.zeros((0, n_features)))
        save_data_array(os.path.join(client_dir, "valid_target"), np.zeros(0))
    
    print(f"Data saved to {folder}")
    return folder

if __name__ == "__main__":
    fire.Fire(prepare_blobs)