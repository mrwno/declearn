# coding: utf-8

import os
import numpy as np
from typing import Tuple

__all__ = ["load_blobs"]

def load_blobs(
    client_id: int,
    folder: str,
    train: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Load blob data for a specific client."""
    # Determine data type prefix
    prefix = "train" if train else "valid"
    
    # Construct file paths
    data_file = os.path.join(folder, f"client_{client_id}", f"{prefix}_data.npy")
    target_file = os.path.join(folder, f"client_{client_id}", f"{prefix}_target.npy")
    
    # Load data
    X = np.load(data_file)
    y = np.load(target_file)
    
    return X, y

def load_all_blobs(
    folder: str,
    train: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Load blob data for all clients."""
    X_list, y_list = [], []
    client_dirs = [d for d in os.listdir(folder) if d.startswith("client_")]
    
    for client_dir in client_dirs:
        client_id = int(client_dir.split("_")[1])
        X, y = load_blobs(client_id, folder, train)
        X_list.append(X)
        y_list.append(y)
    
    return np.vstack(X_list), np.concatenate(y_list)