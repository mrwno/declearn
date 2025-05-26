# test_kmeans.py
import numpy as np
import pytest
from declearn.model.kmeans import FederatedKMeansModel
from declearn.model.api import Vector
import os
from declearn.test_utils import make_importable
with make_importable(os.path.dirname(__file__)):
    from model_testing import ModelTestCase, ModelTestSuite


class TestFKMeansModel(ModelTestSuite):
    # Regular tests for FederatedKMeansModel
    @pytest.fixture
    def model(self):
        model = FederatedKMeansModel(n_clusters=3, random_state=42)
        model.initialize({"features_shape": (2,)})
        return model
    
    @pytest.fixture
    def sample_centroids(self):
        return np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    
    # Adapted from the original test_kmeans.py file
    """ Test ok"""
    def test_fkm_client_normal(self):
        ######## Test 1 : Kmeans classique coté client ######## OK !
        cluster1 = np.array([[0], [1], [3]]) # Centroid 1 = [1.33]
        cluster2 = np.array([[5], [6], [9], [10]]) # Centroid 2 = [7.5]
        data = np.vstack([cluster1, cluster2])
        data = data.astype(np.float64) 

        model2 = FederatedKMeansModel(
            n_clusters=2,
            random_state=42
        )
        model2.initialize({"features_shape": (1,)})

        coefs = {"centroids": np.array([[0.0], [5.0]])}
        vector = Vector.build(coefs)
        model2.set_weights(vector) # Initialisation des centroïdes 
        client_result = model2.compute_kmeans(data, None, client=True)
        # Verif que compute_kmeans fonctionne, supposés être 0.5 et 6.6
        assert np.allclose(np.sort(client_result["centroids"], axis=0), np.array([[0.5], [6.6]]), rtol=1e-5)
        assert np.allclose(np.sort(client_result["counts"], axis=0), np.array([3, 4]), rtol=1e-5)
        
        # Appliquer les updates
        model2.apply_updates(Vector.build(client_result))
        assert np.allclose(np.sort(model2.centroids, axis=0), np.array([[0.5], [6.6]]), rtol=1e-5)
        
        # Refait kmeans pour voir si ça converge
        # Supposés être 1.33 et 7.5 (ça a convergé)
        client_result = model2.compute_kmeans(data, None, client=True)
        assert np.allclose(np.sort(client_result["centroids"]), np.array([[1.33333333], [7.5]]), rtol=1e-5)
        assert np.allclose(np.sort(client_result["counts"]), np.array([3, 4]), rtol=1e-5)
    
    """ Test ok"""
    def test_fkm_client_nb_cluster_less_k(self):
        ###### Test 2 : Kmeans client mais nb_cluster < k ######## OK !
        cluster1 = np.array([[0], [1], [3]]) # Centroid 1 = [1.33]
        cluster2 = np.array([[5], [6], [9], [10]]) # Centroid 2 = [7.5]
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
        assert client_result["centroids"].shape[0] == 2
        assert np.allclose(np.sort(client_result["centroids"], axis=0), np.array([[0.5], [6.6]]), rtol=1e-5)
        assert np.allclose(np.sort(client_result["counts"], axis=0), np.array([3, 4]), rtol=1e-5)
    
    """ Test ok"""
    def test_fkm_server_normal(self):
        ###### Test 3 : Kmeans classique (poids = 1) coté serveur ######## OK !
        cluster1 = np.array([[0], [1], [2]]) # Centroid 1 = [1.]
        cluster2 = np.array([[7], [8], [9], [10]]) # Centroid 2 = [7.5]
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
    
    """ Test ok"""
    def test_fkm_server_weights(self):
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

    def test_model_initialization(self, model):
        assert model.n_clusters == 3
        assert model.random_state == 42
        model.set_weights(Vector.build({"centroids": np.zeros((3, 2))}))    
        assert model.centroids.shape == (3, 2)

    def test_set_weights(self, model, sample_centroids):
        vector = Vector.build({"centroids": sample_centroids})
        model.set_weights(vector)
        assert np.array_equal(model.centroids, sample_centroids)
        assert model.n_clusters == len(sample_centroids)

    def test_set_weight_model_uninitialized(self):
        model = FederatedKMeansModel(n_clusters=3, random_state=42)
        with pytest.raises(ValueError) as exc:
            model.set_weights(Vector.build({"centroids": np.array([[1.0, 2.0]])}))
        assert "Model must be initialized before use." in str(exc.value)

    def test_set_weights_invalid_shape(self, model):
        with pytest.raises(ValueError) as exc:
            model.set_weights(Vector.build({"centroids": np.array([[1.0]])}))
        assert "Centroid shape does not match" in str(exc.value)
    
    
    def test_get_weights(self, model, sample_centroids):
        expected = sample_centroids.copy()
        vector = Vector.build({"centroids": expected})
        model.set_weights(vector)
        weights = model.get_weights()
        assert isinstance(weights, Vector)
        assert "centroids" in weights.coefs
        assert np.array_equal(weights.coefs["centroids"], expected)
    
    def test_weights_independence(self, model, sample_centroids):
        original = sample_centroids.copy()
        vector = Vector.build({"centroids": original})
        model.set_weights(vector)
        weights = model.get_weights()
        weights.coefs["centroids"][0][0] = 99.0
        assert model.centroids[0][0] == original[0][0] 

    def test_compute_batch_predictions(self, test_case):
        """Désactive le test de prédictions pour K-means"""
        pytest.skip("Utilisation de kmeans de sklearn pour les prédictions, consideré comme correct")

    def test_loss_function(self, test_case):
        """Désactive le test de fonction de perte pour K-means"""
        pytest.skip("Non applicable pour K-means")

    def test_serialize_gradients(self, test_case):
        """Désactive le test de sérialisation pour K-means"""
        pytest.skip("Non applicable pour K-means")

    def test_apply_updates(self, test_case):
        """Désactive le test d'application des updates"""
        pytest.skip("update = set_weights pour K-means")

    # Implement required fixtures for ModelTestSuite
    @pytest.fixture
    def test_case(self):
        """Fixture requise par ModelTestSuite"""
        class KMeansTestCase:
            vector_cls = Vector
            framework = "numpy"
            tensor_cls = np.ndarray  
    
            @property
            def dataset(self):
                # Retourner uniquement les features sans labels/weights
                return [(np.array([[0.0], [1.0], [2.0]]),)]

            @property
            def model(self):
                model = FederatedKMeansModel(n_clusters=2)
                model.initialize({"features_shape": (1,)})
                return model

            def assert_correct_device(self, vector):
                pass  # No device management for K-means

        return KMeansTestCase()
