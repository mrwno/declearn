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

from sklearn.cluster import KMeans

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
        privacy_threshold: int = 2,
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
        self.privacy_threshold = privacy_threshold
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)

        self.centroids = np.empty((0,))
        self._previous_centroids = None
        self.init_method = 'k-means++'
        self.max_iter = 300 # Default value from sklearn
        self._kmeans = self._init_kmeans()
        

    @property
    def required_data_info(self) -> Set[str]:
        """Set of required data features metadata keys."""
        return {"features_shape"}

    def initialize(self, data_info: Dict[str, Any]) -> None:
        """Initialize model with aggregated data information."""
        self.features_shape = data_info["features_shape"]
                
    def get_weights(self, trainable: bool = False) -> Vector:
        """Return current centroids as model weights."""
        return Vector.build({"centroids":self.centroids.copy()})
    
    def get_model_centroids(self) -> np.ndarray:
        return self._kmeans.cluster_centers_
    
    def set_weights(self, weights: Vector, trainable: bool = False) -> None:
        """Update model weights (centroids) with provided values."""
        centroids = weights.coefs["centroids"].copy()

        if not hasattr(self, "features_shape"):
            raise ValueError("Model must be initialized before use.")
        
        # Vérifier la cohérence des dimensions
        for centroid in centroids:
            if centroid.shape != self.features_shape:
                raise ValueError(
                    f"Centroid shape does not match  ({centroid.shape})"
                    f"initialized features shape {self.features_shape}"
                )
        self.centroids = centroids
        self.n_clusters = len(self.centroids)   
        self.init_method = centroids
        self._kmeans = self._init_kmeans()

    def _init_kmeans(self) -> KMeans:
        """Create a KMeans instance with the specified parameters."""
        return KMeans(
            n_clusters=self.n_clusters,
            init=self.init_method,
            max_iter=self.max_iter,
            random_state=self.random_state,
            n_init = 1,
        )

    def compute_kmeans(self, data: Any, weights: Any, client: bool) -> Dict[str, np.ndarray]:
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
        print("Initial centroids:", self.centroids)
        if client:
            if self.centroids.size > 0: 
                distances = np.linalg.norm(data[:, np.newaxis] - self.centroids, axis=2)
                assign_cluser = np.argmin(distances, axis=1)
                unique_labels = np.unique(assign_cluser)
                centroids = self.centroids[unique_labels]
                self.n_clusters = len(centroids)
                self.init_method = centroids
            else:
                self.init_method = 'k-means++'
            self.max_iter = 1 
            self._kmeans = self._init_kmeans()
            self._kmeans.fit(data)

            centroids = self._kmeans.cluster_centers_
            labels = self._kmeans.labels_
            counts = np.bincount(labels, minlength=self.n_clusters)
            mask = counts >= self.privacy_threshold
            print("Updated centroids:", centroids[mask])
            return {
                "centroids": centroids[mask],
                "counts": counts[mask]
            }

        else:
            self.max_iter = 300
            self._kmeans = KMeans(n_clusters=self.n_clusters, init='k-means++', random_state=self.random_state)
            self._kmeans.fit(data, sample_weight=weights)
            centroids = self._kmeans.cluster_centers_
            counts = np.ones(self.n_clusters)

            print("Updated centroids:", centroids)
            return {
                "centroids": centroids,
                "counts": counts
            }

 

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
        
        return self._kmeans.predict(batch)
    

    def compute_batch_gradients(self, batch: Any, max_norm: Optional[float] = None) -> Vector:
        """Wrapper for compute_kmeans."""
        features = batch[0]
        result = self.compute_kmeans(features, None, client=True)        
    
        return Vector.build({
            "centroids": result["centroids"]
        })
    
    def apply_updates(self, updates: Vector) -> None:
        self._previous_centroids = self.centroids.copy()
        new_centroids = updates.coefs["centroids"].copy()
        self.n_clusters = len(new_centroids)
        self.centroids = new_centroids
        self.init_method = new_centroids
        self._kmeans = self._init_kmeans()

    def loss_function(self, y_true: Any, y_pred: Any) -> Any:
        raise NotImplementedError("Not used in K-means")
    
    @property
    def device_policy(self) -> Any:
        return "cpu"

    def update_device_policy(self, device: str) -> None:
        pass

    @property
    def converged(self) -> bool:
        """Check if convergence criteria are met (delta < tolerance)."""
        try:
            delta = np.max(np.linalg.norm(self.centroids - self._previous_centroids, axis=1))
            return delta < 1e-4  # Tolérance par défaut
        except AttributeError:
            return False

    def get_config(self) -> Dict[str, Any]:
        """Return model configuration as a dictionary."""
        return {
            "n_clusters": self.n_clusters,
            "privacy_threshold": self.privacy_threshold,
            "random_state": self.random_state,
            "centroids": self.centroids.tolist(),
            "init_method": self.init_method,
            "max_iter": self.max_iter
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "FederatedKMeansModel":
        """Instantiate model from configuration dictionary."""
        model = cls(
            n_clusters=config["n_clusters"],
            privacy_threshold=config["privacy_threshold"],
            random_state=config["random_state"]
        )
        
        # Restaurer les paramètres supplémentaires
        model.init_method = config["init_method"]
        model.max_iter = config["max_iter"]
        
        # Re-créer le modèle KMeans avec ces paramètres
        model._init_kmeans()
        
        if config["centroids"] is not None:
            model.centroids = np.array(config["centroids"])
            model._kmeans.cluster_centers_ = model.centroids
            
        return model