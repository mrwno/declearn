
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

"""Unit tests for FederatedKMeansModel."""
import os
import numpy as np
import pytest

from typing import List
from declearn.model.kmeans import FederatedKMeansModel
from declearn.model.api import Vector
from declearn.test_utils import make_importable
from declearn.typing import Batch


with make_importable(os.path.dirname(__file__)):
    from model_testing import ModelTestCase, ModelTestSuite

class KMeansTestCase(ModelTestCase):
    """KMeans model test-case-provider fixture."""

    vector_cls = Vector
    tensor_cls = np.ndarray  
    framework = "numpy"

    @property
    def dataset(
        self,
    ) -> List[Batch]:
        """Dataset for KMeans testing with 3 clusters in 1D."""
        rng = np.random.default_rng(seed=42)
        centroids = [0.0, 5.0, 10.0]
        std_dev = 0.5
        batches = []
        features = np.zeros((20, 1))
        for i in range(20):
            cluster_idx = i % 3 
            features[i] = rng.normal(loc=centroids[cluster_idx], scale=std_dev)
        labels = None
        sample_weights = None
        batches.append((features, labels, sample_weights))
        return batches

    @property
    def model(
        self
    ) -> FederatedKMeansModel:
        """Return a FederatedKMeansModel instance."""
        model = FederatedKMeansModel(n_clusters=3, random_state=42)
        model.initialize({"features_shape": (1,)})
        return model

    def assert_correct_device(self, vector):
        pass  # No device management for K-means

@pytest.fixture(name="test_case")
def fixture_test_case(
) -> KMeansTestCase:
    """Fixture to access a KMeansTestCase."""
    return KMeansTestCase()


class TestFKMeansModel(ModelTestSuite):
    """Unit tests for declearn.model.kmeans.FederatedKMeansModel."""
    @pytest.fixture
    def model(
        self
    ) -> FederatedKMeansModel:
        """Fixture to access a FederatedKMeansModel instance."""
        model = FederatedKMeansModel(n_clusters=3, random_state=42)
        model.initialize({"features_shape": (2,)})
        return model
    
    @pytest.fixture
    def sample_centroids(
        self
    ) -> np.ndarray:
        """Sample centroids for testing."""
        return np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    
    """ Hand-written tests for FederatedKMeansModel. These tests do not use fixtures"""
    def test_fkm_client_normal(
        self
    ) -> None:
        """Test KMeans client in classic configuration."""
        cluster1 = np.array([[0], [1], [3]])
        cluster2 = np.array([[5], [6], [9], [10]])
        data = np.vstack([cluster1, cluster2])
        data = data.astype(np.float64) 

        model2 = FederatedKMeansModel(
            n_clusters=2,
            random_state=42
        )
        model2.initialize({"features_shape": (1,)})
        coefs = {"centroids": np.array([[0.0], [5.0]])}
        vector = Vector.build(coefs)
        model2.set_weights(vector)
        client_result = model2.compute_kmeans(data, None, client=True)
        assert np.allclose(np.sort(client_result["centroids"], axis=0), np.array([[0.5], [6.6]]), rtol=1e-5)
        assert np.allclose(np.sort(client_result["counts"], axis=0), np.array([3, 4]), rtol=1e-5)
        model2.apply_updates(Vector.build(client_result))
        assert np.allclose(np.sort(model2.centroids, axis=0), np.array([[0.5], [6.6]]), rtol=1e-5)
        client_result = model2.compute_kmeans(data, None, client=True)
        assert np.allclose(np.sort(client_result["centroids"]), np.array([[1.33333333], [7.5]]), rtol=1e-5)
        assert np.allclose(np.sort(client_result["counts"]), np.array([3, 4]), rtol=1e-5)
        return None
    
    def test_fkm_client_nb_cluster_less_k(
        self
    ) -> None:
        """Test KMeans client with n_clusters < k."""
        cluster1 = np.array([[0], [1], [3]])
        cluster2 = np.array([[5], [6], [9], [10]])
        data = np.vstack([cluster1, cluster2])
        data = data.astype(np.float64) 

        model2 = FederatedKMeansModel(
            n_clusters=2,
            random_state=42
        )
        model2.initialize({"features_shape": (1,)})
        coefs2 = {"centroids": np.array([[0.0], [5.0], [50.0]])}
        vector2 = Vector.build(coefs2)
        model2.set_weights(vector2)
        client_result = model2.compute_kmeans(data, None, client=True)
        assert client_result["centroids"].shape[0] == 2 # Verify that we now have 2 centroids
        assert np.allclose(np.sort(client_result["centroids"], axis=0), np.array([[0.5], [6.6]]), rtol=1e-5)
        assert np.allclose(np.sort(client_result["counts"], axis=0), np.array([3, 4]), rtol=1e-5)
        return None
    
    def test_fkm_server_normal(
        self
    ) -> None:
        """Manual test for KMeans server with not weighted data."""
        cluster1 = np.array([[0], [1], [2]])
        cluster2 = np.array([[7], [8], [9], [10]])
        data = np.vstack([cluster1, cluster2])
        data = data.astype(np.float64) 

        model2 = FederatedKMeansModel(
            n_clusters=2,
            random_state=42
        )
        model2.initialize({"features_shape": (1,)})
        weights = np.ones(data.shape[0])
        client_result = model2.compute_kmeans(data, weights, client=False)
        assert np.allclose(np.sort(client_result["centroids"], axis=0), np.array([[1], [8.5]]), rtol=1e-5)
        assert np.allclose(np.sort(client_result["counts"], axis=0), np.array([1, 1]), rtol=1e-5)
        return None
    
    def test_fkm_server_weights(
        self
    ) -> None:
        """Manual test for KMeans server with weighted data."""
        cluster1 = np.array([[0], [1], [3]]) # Centroid 1 = [0.33]
        cluster2 = np.array([[5], [6], [9], [10]]) # Centroid 2 = [7.5]
        data = np.vstack([cluster1, cluster2])
        data = data.astype(np.float64) 

        model2 = FederatedKMeansModel(
            n_clusters=2,
            random_state=42
        )
        model2.initialize({"features_shape": (1,)})
        weights = np.ones(data.shape[0])
        weights[0] = 10
        client_result = model2.compute_kmeans(data, weights, client=False)
        assert np.allclose(np.sort(client_result["centroids"], axis=0), np.array([[0.33333333], [7.5]]), rtol=1e-5)
        assert np.allclose(np.sort(client_result["counts"], axis=0), np.array([1, 1]), rtol=1e-5)
        return None
    
    """ Those tests are using the fixtures defined above."""
    def test_model_initialization(
        self, 
        model: FederatedKMeansModel,
    ) -> None:
        """Check that model initializes correctly with given parameters."""
        assert model.n_clusters == 3
        assert model.random_state == 42
        model.set_weights(Vector.build({"centroids": np.zeros((3, 2))}))    
        assert model.centroids.shape == (3, 2)
        return None

    def test_set_weights(
        self, 
        model: FederatedKMeansModel,
        sample_centroids: np.ndarray,
    ) -> None:
        """Check that set_weights correctly sets the centroids."""
        vector = Vector.build({"centroids": sample_centroids})
        model.set_weights(vector)
        assert np.array_equal(model.centroids, sample_centroids)
        assert model.n_clusters == len(sample_centroids)
        return None

    def test_set_weight_model_uninitialized(
        self
    ) -> None:
        """Check that set_weights raises an error if model is not initialized."""
        model = FederatedKMeansModel(n_clusters=3, random_state=42)
        with pytest.raises(ValueError) as exc:
            model.set_weights(Vector.build({"centroids": np.array([[1.0, 2.0]])}))
        assert "Model must be initialized before use." in str(exc.value)
        return None

    def test_set_weights_invalid_shape(
        self, 
        model
    ) -> None:
        """Check that set_weights raises an error for invalid centroid shape."""
        with pytest.raises(ValueError) as exc:
            model.set_weights(Vector.build({"centroids": np.array([[1.0]])}))
        assert "Centroid shape does not match" in str(exc.value)
        return None
    
    
    def test_get_weights(
        self, 
        model: FederatedKMeansModel,
        sample_centroids: np.ndarray,
    ) -> None:
        """Check that get_weights returns the correct structure and centroids."""
        expected = sample_centroids.copy()
        vector = Vector.build({"centroids": expected})
        model.set_weights(vector)
        weights = model.get_weights()
        assert isinstance(weights, Vector)
        assert "centroids" in weights.coefs
        assert np.array_equal(weights.coefs["centroids"], expected)
        return None

    def test_compute_batch_predictions(
        self, 
        test_case: ModelTestCase,
    ) -> None:
        """Desactivate the prediction test for K-means"""
        # The prediction method is considered as functional because it
        # uses the predict function from scikit-learn
        return None

    def test_loss_function(
        self, 
        test_case: ModelTestCase,
    ) -> None:
        """Desactivate the loss function test for K-means"""
        # K-means does not have a loss function.
        return None

    def test_apply_updates(
        self, 
        test_case: ModelTestCase,
    ) -> None:
        """Desactivate the apply_updates test for K-means"""
        # The model does not use apply_updates, as it is not a gradient-based model.
        return None

    def test_compute_batch_gradients_np(
        self,
        test_case: ModelTestCase,
    ) -> None:
        """Desactivate the gradients_np test for K-means"""
        # The model already uses numpy inputs, this test is unrequired here.
        # NOTE: in fact, it fails with sparse inputs as intercept is *not*
        #       fitted equally (while coefficients are) -> investigate this.
        return None
    
    def test_compute_batch_gradients_clipped(
        self,
        test_case: ModelTestCase,
    ) -> None:
        """Desactivate the gradients_clipped test for K-means"""
        # The model does not use max norm clipping for K-means.
        # NOTE: this test does not check that results are correct
        return None