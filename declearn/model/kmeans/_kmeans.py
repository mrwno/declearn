# coding: utf-8

# Copyright 2023 Inria (Institut National de Recherche en Informatique
# et Automatique)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Model subclass for federated K-means clustering."""

from typing import Any, Dict, Optional, Set
import numpy as np
from declearn.model.api import Model
from declearn.data_info import aggregate_data_info
from declearn.utils import register_type

__all__ = ["FederatedKMeansModel"]

@register_type(name="FederatedKMeansModel", group="Model")
class FederatedKMeansModel(Model):
    """Federated K-means clustering model with convergence tracking.

    This `Model` subclass provides a federated implementation of the
    K-means clustering algorithm, supporting centroid initialization,
    client-side cluster assignment, and server-side centroid updates.

    Notes regarding device management (CPU, GPU, etc.):

    - This Model operates purely on CPU and is unaffected by device-
      management policies.
    - Calling the `update_device_policy` method has no effect, and
      raises a UserWarning if a GPU-targetting policy is passed to
      it directly.
    """

    def __init__(
        self,
        n_clusters: int,
        init_method: str = "random",
        tol: float = 1e-4,
        random_state: Optional[int] = None,
    ):
        """Instantiate a federated K-means clustering model.

        Parameters
        ----------
        n_clusters : int
            The number of clusters to find during the federated K-means training.
        init_method : str, default="random"
            Method used to initialize centroids. Only "random" is currently supported.
            "k-means++" is listed as an option but not yet implemented.
        tol : float, default=1e-4
            Convergence tolerance. The training stops when the maximum
            change in centroid positions is below this threshold.
        random_state : int or None
            Seed used for random number generation to ensure reproducibility
            during centroid initialization.

        Raises
        ------
        ValueError
            If an unsupported initialization method is provided.
        """
        super().__init__()
        self.n_clusters = n_clusters
        self.init_method = init_method
        self.tol = tol
        self.random_state = random_state
        self.centroids = None  # type: Optional[np.ndarray]
        self.delta = np.inf    # Track maximum centroid change between iterations

        # Validate the initialization method
        if init_method not in ["random", "k-means++"]:
            raise ValueError(f"Unsupported initialization method: {init_method}")

        # Set the seed if provided
        if random_state is not None:
            np.random.seed(random_state)


    @property
    def required_data_info(self) -> Set[str]:
        """Set of required data features metadata keys."""
        return {"features_shape"}

    def initialize(self, data_info: Dict[str, Any]) -> None:
        """Initialize centroids based on aggregated data features metadata.
        
        Parameters
        ----------
        data_info : dict
            Aggregated data features metadata containing "features_shape"
        """
        data_info = aggregate_data_info([data_info], self.required_data_info)
        n_features = data_info["features_shape"][0]
        
        if self.init_method == "random":
            # Initialize random centroids within [0, 1) range
            self.centroids = np.random.rand(self.n_clusters, n_features)
        elif self.init_method == "k-means++":
            # Not yet implemented
            raise NotImplementedError("K-means++ initialization is not yet implemented.")

    def get_weights(self, trainable: bool = False) -> np.ndarray:
        """Return current centroids as model weights."""
        return self.centroids.copy() if self.centroids is not None else None
    
    def set_weights(self, weights: np.ndarray, trainable: bool = False) -> None:
        """Update model weights (centroids) with provided values."""
        self.centroids = weights.copy()

    def compute_batch_gradients(self, batch: Any, max_norm: Optional[float] = None) -> Dict[str, np.ndarray]:
        """Compute cluster statistics (sums and counts) for the **entire dataset** 
        (no batches), as required by Swier Garst's paper.
        Corresdponds to line 17 of the FKM paper.
        Parameters
        ----------
        batch : tuple
            Input data batch containing features as first element
            **Note:** This method processes the **entire dataset at once** 
            (not batches) to comply with the paper's algorithm.
        max_norm : float or None, optional
            Unused parameter (maintained for API compatibility)

        Returns
        -------
        dict
            Contains "centroids" (updated cluster centers) and "counts" (cluster sizes).
        """
        X = batch[0]  # Input features shape=(batch_size, n_features)
        
        # 1. Compute distances between samples and centroids
        distances = np.linalg.norm(X[:, None] - self.centroids, axis=2)
        
        # 2. Assign samples to nearest clusters
        labels = np.argmin(distances, axis=1)
        
        # 3. Calculate cluster sums and counts
        sums = np.zeros_like(self.centroids)
        counts = np.zeros(self.n_clusters)
        
        for k in range(self.n_clusters):
            mask = (labels == k)
            sums[k] = np.sum(X[mask], axis=0)
            counts[k] = np.sum(mask)

        counts_safe = counts[:, None] + 1e-8  
        centroids = sums / counts_safe
        # Return S_i and C_i from FKM paper
        return {"centroids": centroids, "counts": counts}

    def apply_updates(
        self,
        C_l: np.ndarray,  # Pre-concatenated list of all local centroids (shape: [M, n_features])
        S_l: np.ndarray,   # Pre-concatenated list of corresponding cluster weights (shape: [M])
    ) -> None:
        """Perform weighted K-means clustering on pre-aggregated client centroids.
        
        Parameters
        ----------
        C_l : np.ndarray
            All client centroids concatenated into a single array.
            Each row represents a centroid from any client.
        S_l : np.ndarray
            Corresponding cluster weights (typically sample counts) concatenated.
            Used to weight centroids during aggregation.
        """
        
        current_centroids = self.centroids.copy() if self.centroids is not None else None
        delta = np.inf

        if len(C_l) != len(S_l):
            raise ValueError("C_l and S_l must have same number of elements")
        if len(C_l) == 0:
            return

        # While loop until convergence
        while delta >= self.tol:
            # Calculate disances
            distances = np.linalg.norm(C_l[:, None] - current_centroids, axis=2)
            # Assign each centroid to the nearest global centroid
            assignments = np.argmin(distances, axis=1)
            
            # Update centroids
            new_centroids = np.zeros_like(current_centroids)
            for k in range(self.n_clusters):
                mask = (assignments == k)
                if np.sum(mask) == 0:
                    new_centroids[k] = current_centroids[k]
                    continue
                weights = S_l[mask]
                weighted_sum = np.sum(C_l[mask] * weights[:, None], axis=0)
                total_weight = np.sum(weights)
                new_centroids[k] = weighted_sum / total_weight

            delta = np.max(np.linalg.norm(new_centroids - current_centroids, axis=1))
            current_centroids = new_centroids

        self.centroids = current_centroids
        self.delta = delta

    def compute_batch_predictions(self, batch: Any) -> np.ndarray:
        """Predict cluster assignments for a batch of data.
        
        Parameters
        ----------
        batch : tuple
            Input data batch containing features as first element

        Returns
        -------
        np.ndarray
            Cluster indices for each sample in the batch
        """
        X = batch[0]
        distances = np.linalg.norm(X[:, None] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def local_kmeans_iteration(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Perform one local K-Means iteration: assign data points to current centroids,
        remove empty clusters, and run one update step using non-empty centroids.

        Parameters
        ----------
        X : np.ndarray
            Local client data, shape = (n_samples, n_features)

        Returns
        -------
        dict
            "centroids": np.ndarray of updated cluster centers (shape = (k', n_features))
            "counts": np.ndarray of sample counts per cluster (shape = (k',))
        """
        if self.centroids is None:
            raise ValueError("Centroids must be initialized before running local iteration.")

        # line 14 of the FKM paper
        distances = np.linalg.norm(X[:, None] - self.centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        # lines 15-16 of the FKM paper
        new_centroids = []
        new_counts = []
        for i in range(self.n_clusters):
            mask = labels == i
            count = np.sum(mask)
            if count > 0:
                new_centroids.append(np.mean(X[mask], axis=0))
                new_counts.append(count)
        
        new_centroids = np.array(new_centroids)
        new_counts = np.array(new_counts)

        
        self.centroids = new_centroids
        self.n_clusters = len(new_centroids)

        return {"centroids": new_centroids, "counts": new_counts}


    @property
    def converged(self) -> bool:
        """Check if convergence criteria are met (delta < tolerance)."""
        return self.delta < self.tol

    def get_config(self) -> Dict[str, Any]:
        """Return model configuration as a dictionary."""
        return {
            "n_clusters": self.n_clusters,
            "init_method": self.init_method,
            "tol": self.tol,
            "random_state": self.random_state,
            "centroids": self.centroids.tolist() if self.centroids is not None else None,
            "delta": float(self.delta),
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "FederatedKMeansModel":
        """Instantiate model from configuration dictionary."""
        model = cls(
            n_clusters=config["n_clusters"],
            init_method=config["init_method"],
            tol=config["tol"],
            random_state=config["random_state"],
        )
        if config["centroids"] is not None:
            model.centroids = np.array(config["centroids"])
        model.delta = config["delta"]
        return model