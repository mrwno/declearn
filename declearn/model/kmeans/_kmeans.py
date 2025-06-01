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

import numpy as np
from typing import Any, Dict, Optional, Set, Union
from declearn.model.api import Model
from declearn.model.api import Vector
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
        privacy_threshold : int = 2
            Minimum number of samples required in a cluster to be considered valid.
            Clusters with fewer samples will be ignored during updates.
            This is used to ensure privacy by preventing small clusters from being
            reported back to the server. Value is set to 2 by default.
        random_state : int or None
            Seed used for random number generation to ensure reproducibility
            during centroid initialization.

        Raises
        ------
        ValueError
            If an unsupported initialization method is provided.
        """
        super().__init__(model=None)
        self.n_clusters = n_clusters # type: int
        self.privacy_threshold = privacy_threshold # type: int
        self.random_state = random_state # type: Optional[int]
        if random_state is not None:
            np.random.seed(random_state)

        self.centroids = np.empty((0,)) # type: np.ndarray
        self._previous_centroids = None # type: Optional[np.ndarray]
        self.init_method = 'k-means++' # type: Union[str, np.ndarray]
        self.max_iter = 300  # type: int
        self._kmeans = self._init_kmeans() # type: KMeans
        

    @property
    def required_data_info(
        self
    ) -> Set[str]:
        """Set of required data features metadata keys."""
        return {"features_shape"}

    def initialize(
        self, 
        data_info: Dict[str, Any]
    ) -> None:
        """Initialize model with aggregated data information."""
        self.features_shape = data_info["features_shape"]
                
    def get_weights(
        self, 
        trainable: bool = False # Not used in KMeans, APi compatibility
    ) -> Vector:
        """Return current centroids as model weights."""
        return Vector.build({"centroids":self.centroids.copy()})
    
    def set_weights(
        self, 
        weights: Vector, 
        trainable: bool = False # Not used in KMeans, API compatibility
    ) -> None:
        """Update model weights (centroids) with provided values."""
        centroids = weights.coefs["centroids"].copy()
        if not hasattr(self, "features_shape"):
            raise ValueError("Model must be initialized before use.")
        # Check if the shape of centroids matches the expected features shape
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

    def _init_kmeans(
        self
    ) -> KMeans:
        """Create a KMeans instance with the specified parameters."""
        return KMeans(
            n_clusters=self.n_clusters,
            init=self.init_method,
            max_iter=self.max_iter,
            random_state=self.random_state,
            n_init = 1,
        )

    def compute_kmeans(
        self, 
        data: np.ndarray,
        weights: np.ndarray, 
        client: bool,
    ) -> Dict[str, np.ndarray]:
        """ Compute K-means clustering on the provided data.
        Parameters
        ----------
        data : np.ndarray
            Input data for clustering, shape (n_samples, n_features).
        weights : np.ndarray
            Sample weights for each data point, shape (n_samples,).
        client : bool
            If True, perform client-side clustering; otherwise, server-side.
        Returns
        -------
        dict
            Contains "centroids" (updated cluster centers) and "counts" (cluster sizes).
        """
        if client:
            if self.centroids.size > 0: 
                self._kmeans.fit(self.centroids)
                assign = self._kmeans.predict(data)
                unique_labels = np.unique(assign)
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
            return {
                "centroids": centroids,
                "counts": counts
            }

 

    def compute_batch_predictions(
        self, 
        batch: np.ndarray
    ) -> np.ndarray:
        """Predict cluster assignments for a batch of data.
        
        Parameters
        ----------
        batch : np.ndarray
            Contains the data for which to predict cluster assignments,

        Returns
        -------
        np.ndarray
            Cluster indices for each sample in the batch
        """
        
        return self._kmeans.predict(batch)
    

    def compute_batch_gradients(
        self, 
        batch: np.ndarray, 
        max_norm: Optional[float] = None # Not used in KMeans, API compatibility
    ) -> Vector:
        """Wrapper for compute_kmeans."""
        features = batch[0]
        result = self.compute_kmeans(features, None, client=True)        
    
        return Vector.build({
            "centroids": result["centroids"]
        })
    
    def apply_updates(
        self, 
        updates: Vector
    ) -> None:
        """Apply updates to the model's centroids."""
        self._previous_centroids = self.centroids.copy()
        new_centroids = updates.coefs["centroids"].copy()
        self.n_clusters = len(new_centroids)
        self.centroids = new_centroids
        self.init_method = new_centroids
        self._kmeans = self._init_kmeans()

    def loss_function(
        self, 
        y_true: Any, 
        y_pred: Any
    ) -> Any:
        raise NotImplementedError("Not used in K-means")
    
    @property
    def device_policy(
        self
    ) -> Any:
        return "cpu"

    def update_device_policy(
        self, 
        device: str
    ) -> None:
        pass

    @property
    def converged(
        self
    ) -> bool:
        """Check if convergence criteria are met (delta < 1e-4 )."""
        try:
            delta = np.max(np.linalg.norm(self.centroids - self._previous_centroids, axis=1))
            return delta < 1e-4 
        except AttributeError:
            return False

    def get_config(
        self
    ) -> Dict[str, Any]:
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
    def from_config(
        cls, 
        config: Dict[str, Any]
    ) -> "FederatedKMeansModel":
        """Instantiate FederatedKmeansModel from configuration dictionary."""
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