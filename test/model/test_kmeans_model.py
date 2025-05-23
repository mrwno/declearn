# test_kmeans.py
import numpy as np
import pytest
from declearn.model.kmeans import FederatedKMeansModel
from declearn.model.api import Vector

class TestFKMeansModel:
    """ Test ok"""
    def test_fkm_client_normal(self):
        ######## Test 1 : Kmeans classique coté client ######## OK !
        cluster1 = np.array([[0], [1], [3]]) # Centroid 1 = [1.33]
        cluster2 = np.array([[5], [6], [9], [10]]) # Centroid 2 = [7.5]
        data = np.vstack([cluster1, cluster2])
        data = data.astype(np.float64) 

        model = FederatedKMeansModel(
            n_clusters=2,
            random_state=42
        )

        coefs = {"centroids": np.array([[0.0], [5.0]])}
        vector = Vector.build(coefs)
        model.set_weights(vector) # Initialisation des centroïdes 
        # Verif que set_weights fonctionne
        assert np.allclose(np.sort(model.centroids, axis=0), coefs["centroids"], rtol=1e-5)
        client_result = model.compute_kmeans(data, None, client=True)
        # Verif que compute_kmeans fonctionne, supposés être 0.5 et 6.6
        assert np.allclose(np.sort(client_result["centroids"], axis=0), np.array([[0.5], [6.6]]), rtol=1e-5)
        assert np.allclose(np.sort(client_result["counts"], axis=0), np.array([3, 4]), rtol=1e-5)
        # Verif que les centroïdes sont bien mis à jour
        assert np.allclose(np.sort(model.centroids, axis=0), np.sort(client_result["centroids"], axis=0), rtol=1e-5)
        # Verif que la fonction get_weights fonctionne
        assert np.allclose(np.sort(model.get_weights().coefs["centroids"], axis=0), np.sort(client_result["centroids"], axis=0), rtol=1e-5)

        # Refait kmeans pour voir si ça converge
        # Supposés être 1.33 et 7.5 (ça a convergé)
        client_result = model.compute_kmeans(data, None, client=True)
        assert np.allclose(np.sort(client_result["centroids"]), np.array([[1.33333333], [7.5]]), rtol=1e-5)
        assert np.allclose(np.sort(client_result["counts"]), np.array([3, 4]), rtol=1e-5)
    
    """ Test ok"""
    def test_fkm_client_nb_cluster_less_k(self):
        ###### Test 2 : Kmeans client mais nb_cluster < k ######## OK !
        cluster1 = np.array([[0], [1], [3]]) # Centroid 1 = [1.33]
        cluster2 = np.array([[5], [6], [9], [10]]) # Centroid 2 = [7.5]
        data = np.vstack([cluster1, cluster2])
        data = data.astype(np.float64) 

        model = FederatedKMeansModel(
            n_clusters=2,
            random_state=42
        )

        coefs2 = {"centroids": np.array([[0.0], [5.0], [50.0]])}
        vector2 = Vector.build(coefs2)


        model.set_weights(vector2)
        assert np.allclose(np.sort(model.centroids, axis=0), np.sort(coefs2["centroids"]), rtol=1e-5)
        client_result = model.compute_kmeans(data, None, client=True)
        assert np.allclose(np.sort(client_result["centroids"], axis=0), np.array([[0.5], [6.6]]), rtol=1e-5)
        assert np.allclose(np.sort(client_result["counts"], axis=0), np.array([3, 4]), rtol=1e-5)
    
    """ Test ok"""
    def test_fkm_server_normal(self):
        ###### Test 3 : Kmeans classique (poids = 1) coté serveur ######## OK !
        cluster1 = np.array([[0], [1], [3]]) # Centroid 1 = [1.33]
        cluster2 = np.array([[5], [6], [9], [10]]) # Centroid 2 = [7.5]
        data = np.vstack([cluster1, cluster2])
        data = data.astype(np.float64) 

        model = FederatedKMeansModel(
            n_clusters=2,
            random_state=42
        )
        weights = np.ones(data.shape[0])
        client_result = model.compute_kmeans(data, weights, client=False)
        assert np.allclose(np.sort(client_result["centroids"], axis=0), np.array([[1.33333], [7.5]]), rtol=1e-5)
        assert np.allclose(np.sort(client_result["counts"], axis=0), np.array([1, 1]), rtol=1e-5)
    
    """ Test ok"""
    def test_fkm_server_weights(self):
        cluster1 = np.array([[0], [1], [3]]) # Centroid 1 = [0.33]
        cluster2 = np.array([[5], [6], [9], [10]]) # Centroid 2 = [7.5]
        data = np.vstack([cluster1, cluster2])
        data = data.astype(np.float64) 

        model = FederatedKMeansModel(
            n_clusters=2,
            random_state=42
        )

        weights = np.ones(data.shape[0])
        weights[0] = 10
        client_result = model.compute_kmeans(data, weights, client=False)
        assert np.allclose(np.sort(client_result["centroids"], axis=0), np.array([[0.33333333], [7.5]]), rtol=1e-5)
        assert np.allclose(np.sort(client_result["counts"], axis=0), np.array([1, 1]), rtol=1e-5)

    
    


