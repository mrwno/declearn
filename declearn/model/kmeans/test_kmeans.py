# test_kmeans.py
import numpy as np
from _kmeans import FederatedKMeansModel
from declearn.model.api import Vector

def test_federated_kmeans():
    ######## Test 1 : Kmeans classique coté client ######## OK !
    cluster1 = np.array([[0, 0], [1, 1], [3, 3]]) # Centroid 1 = [1.33, 1.33]
    cluster2 = np.array([[5, 5], [6, 6], [9, 9], [10, 10]]) # Centroid 2 = [7.5, 7.5]
    data = np.vstack([cluster1, cluster2])
    data = data.astype(np.float64) 

    model = FederatedKMeansModel(
        n_clusters=2,
        init_method="random",
        tol=1e-6,  # Tolérance plus stricte
        random_state=42
    )

    coefs = {"centroid": np.array([[0.0, 0.0], [5.0, 5.0]])}
    vector = Vector.build(coefs)

    coefs2 = {"centroid": np.array([[0.0, 0.0], [5.0, 5.0], [50.0, 50.0]])}
    vector2 = Vector.build(coefs2)

    model.initialize({"features_shape": (2,)})
    print("Test 1 : Kmeans classique coté client")
    model.set_weights(vector) # Initialisation des centroïdes 
    print("Centroides avant le calcul:", model.centroids)
    client_result = model.compute_kmeans(data, None, client=True)
    print("Centroïdes clients:", client_result["centroids"]) #Supposés être 0.5 et 6.6
    print("Poids clients:", client_result["counts"])
    print("Centroides avant le calcul:", model.centroids) # verifier que les centroïdes sont bien mis à jour
    client_result = model.compute_kmeans(data, None, client=True) # 
    print("Centroïdes clients:", client_result["centroids"]) # Supposés être 1.33 et 7.5 (ça a convergé)
    print("Poids clients:", client_result["counts"]) 
    #Pas de changement de centroides
    print("Centroides avant le calcul:", model.centroids)
    client_result = model.compute_kmeans(data, None, client=True)
    print("Centroïdes clients:", client_result["centroids"])
    print("Poids clients:", client_result["counts"]) 

    ###### Test 2 : Kmeans client mais nb_cluster < k ######## OK !
    print("\n\n ############## \n\n Test 2 : Kmeans client mais nb_cluster < k")
    model.set_weights(vector2)
    print("Centroides avant le calcul:", model.centroids)
    client_result = model.compute_kmeans(data, None, client=True)
    print("Centroïdes clients:", client_result["centroids"]) #Supposés être 0.5 et 6.6
    print("Poids clients:", client_result["counts"])

    ###### Test 3 : Kmeans classique (poids = 1) coté serveur ######## OK !
    weights = np.ones(data.shape[0])
    print("\n\n ############## \n\n Test 3 : Kmeans classique coté serveur")
    model.set_weights(vector) 
    print("Centroides avant le calcul:", model.centroids)
    client_result = model.compute_kmeans(data, weights, client=False)
    print("Centroïdes clients:", client_result["centroids"]) #Supposés être 1.33 et 7.5 car convergence
    print("Poids clients:", client_result["counts"])

    ###### Test 4 : Kmeans classique (poids = 1) coté serveur - nb_cluster < k ######## OK !
    weights = np.ones(data.shape[0])
    print("\n\n ############## \n\n Test 4 : Kmeans classique coté serveur avec nb_cluster < k")
    # Test de la fonction avec des centroids initialisés, en temps normal les centroids sont initialisés aléatoirement
    model.set_weights(vector2)
    print("Centroides avant le calcul:", model.centroids)
    client_result = model.compute_kmeans(data, weights, client=False)
    print("Centroïdes clients:", client_result["centroids"]) #Supposés être 1.33, 7.5 et 50.0 car convergence et 50.0 non modifié
    print("Poids clients:", client_result["counts"])

    ###### Test 5 : Kmeans pondéré coté serveur ######## OK ! ######
    weights = np.ones(data.shape[0])
    weights[0] = 10
    print("\n\n ############## \n\n Test 5 : Kmeans pondéré coté serveur")
    # Test de la fonction avec des centroids initialisés, en temps normal les centroids sont initialisés aléatoirement
    model.set_weights(vector2)
    print("Centroides avant le calcul:", model.centroids)
    client_result = model.compute_kmeans(data, weights, client=False)
    print("Centroïdes clients:", client_result["centroids"]) #Supposés être 0.33, 7.5 et 50.0 car convergence et 50.0 non modifié
    print("Poids clients:", client_result["counts"])

    


if __name__ == "__main__":
    test_federated_kmeans()