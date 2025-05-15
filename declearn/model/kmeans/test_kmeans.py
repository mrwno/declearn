# test_kmeans.py
import numpy as np
from _kmeans import FederatedKMeansModel

def test_federated_kmeans():
    # 1. Générer des données synthétiques déterministes
    cluster1 = np.array([[0, 0]] * 50)  # 50 points à (0,0)
    cluster2 = np.array([[5, 5]] * 50)  # 50 points à (5,5)
    data = np.vstack([cluster1, cluster2])

    # 2. Initialiser le modèle
    model = FederatedKMeansModel(
        n_clusters=2,
        init_method="random",
        tol=1e-4,
        random_state=42
    )
    model.initialize({"features_shape": (2,)})
    # Remplacer l'initialisation aléatoire par des centroïdes fixes pour le test
    model.set_centroids(np.array([[0.0, 0.0], [5.0, 5.0]]))

    # 3. Tester le côté client
    client_result = model.compute_kmeans(data, None, client=True)
    print("Centroïdes clients:")
    print(client_result["centroids"])

    # 4. Tester le côté serveur avec un cluster artificiel
    server_data = np.vstack([
        client_result["centroids"], 
        [[10, 10], [10, 10]]  # Ajout de 2 points "bruit"
    ])
    server_weights = np.concatenate([
        client_result["counts"], 
        [5, 5]  # Poids pour les nouveaux points
    ])
    # Après avoir reçu les centroïdes clients
    model.set_centroids(client_result["centroids"])  # [0,0] et [5,5]
    model.compute_kmeans(server_data, server_weights, client=False)
    print("\nCentroïdes finaux après agrégation serveur:")
    print(model.centroids)


if __name__ == "__main__":
    test_federated_kmeans()