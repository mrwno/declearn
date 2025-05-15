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
from declearn.model.api import Vector

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
        super().__init__(model=None)
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

    def get_centroids(self, trainable: bool = False) -> np.ndarray:
        """Return current centroids as model weights."""
        return self.centroids.copy() if self.centroids is not None else None
    
    def set_centroids(self, centroids: np.ndarray, trainable: bool = False) -> None:
        """Update model weights (centroids) with provided values."""
        self.centroids = centroids.copy()

    def compute_kmeans(self, data: Any, weights: Any, client: bool, ) -> Dict[str, np.ndarray]:
        """Compute cluster statistics (sums and counts) for the **entire dataset** 
        ,as required by Swier Garst's paper.
        Parameters
        ----------
        data : tuple
            Input data batch containing features as first element
            **Note:** This method processes the **entire dataset at once** 
            (not batches) to comply with the paper's algorithm.
        Returns
        -------
        dict
            Contains "centroids" (updated cluster centers) and "counts" (cluster sizes).
        """
        if data is None:
            raise ValueError("Data cannot be None.")
        
        X = data

        if client:
            # Client-side: centroids are updated outside of this function (line 13)
            current_centroids = self.centroids.copy() if self.centroids is not None else None
            # Client must have initialized centroids
            if current_centroids is None:
                raise ValueError("Centroids must be initialized before applying updates.")
            # Put weights to 1 for client-side
            weights = np.ones(X.shape[0])
        else:
        # Server-side: need to initialize centroids
            # Choose self.n_clusters random data points as centroids
            indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
            self.centroids = X[indices]
            current_centroids = self.centroids.copy()
            # Check that data and weights have the same number of elements
            if len(data) != len(weights):
                raise ValueError("Parameters 'data' and 'weights' must have the same number of elements.")
            
        delta = np.inf
        # if a centroid is not used
        counts = np.zeros(self.n_clusters)

        while delta >= self.tol:
            # Calculate distances
            distances = np.linalg.norm(data[:, None] - current_centroids, axis=2)
            # Assign each datapoint to the nearest centroid
            labels = np.argmin(distances, axis=1)
            # Update centroids
            new_centroids = np.zeros_like(current_centroids)
            for k in range(self.n_clusters):
                mask = (labels == k)
                counts[k] = np.sum(mask)
                if counts[k] == 0:
                    new_centroids[k] = current_centroids[k]
                    continue
                weights_masked = weights[mask]    
                new_centroids[k] = np.average(X[mask], axis=0, weights=weights_masked)

            delta = np.max(np.linalg.norm(new_centroids - current_centroids, axis=1))
            current_centroids = new_centroids
            if client:
                break

        # Filter out empty clusters for client-side
        if client:
            non_empty = counts > 0
            final_centroids = current_centroids[non_empty]
            final_counts = counts[non_empty]
        else:
            final_centroids = current_centroids
            final_counts = counts

        self.centroids = final_centroids
        # Return C_i and N_i
        return {"centroids": final_centroids, "counts": final_counts}

 

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
    
    #################################

    # Functions required from Model API - AI generated
    # ToDo + change comments from functions
    def get_weights(self, trainable: bool = False) -> Vector:
        return Vector.from_array(self.centroids.copy()) if self.centroids is not None else None
    
    def set_weights(self, weights: Vector, trainable: bool = False) -> None:
        self.centroids = weights.coefs.copy()

    def compute_batch_gradients(self, batch: Any, max_norm: Optional[float] = None) -> Dict[str, Vector]:
        """Wrapper pour compute_kmeans (nécessaire pour l'API DecLearn)."""
        result = self.compute_kmeans(batch[0], None, client=True)
        return {
            "centroids": Vector.from_array(result["centroids"]),
            "counts": Vector.from_array(result["counts"])
        }

    def apply_updates(self, updates: Dict[str, Vector]) -> None:
        """Mise à jour des centroïdes (nécessaire pour l'API DecLearn)."""
        centroids = updates["centroids"].coefs
        self.centroids = centroids.copy()

    # 2. Ajouter des méthodes factices pour le reste de l'API
    def loss_function(self, y_true: Any, y_pred: Any) -> Any:
        raise NotImplementedError("Not used in K-means")
    
    @property
    def device_policy(self) -> Any:
        return "cpu"

    def update_device_policy(self, device: str) -> None:
        pass
    
    #############################


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