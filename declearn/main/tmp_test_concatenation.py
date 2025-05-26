from declearn.model.api import Vector
from declearn import messaging
import numpy as np

# test de concatenation de vecteur
def test():
    client1 = messaging.KMeansInitReply(
        cluster_means=Vector.build({"centroids":  np.array([[1.0, 2.0], [2.0, 3.0]])}),
        sample_counts=[1, 2]
    )
    client2 = messaging.KMeansInitReply(
        cluster_means=Vector.build({"centroids":  np.array([[3.0, 4.0], [4.0, 5.0]])}),
        sample_counts=[3, 4]
    )
    replies = {"client1": client1, "client2": client2}

    all_centroids = []
    all_counts = []

    for reply in replies.values():
        centroids = reply.cluster_means.coefs["centroids"]
        all_centroids.extend(centroids)
        all_counts.extend(reply.sample_counts)

    print("All centroids:", all_centroids)
    vector = Vector.build({"centroids" :np.array(all_centroids)})
    print("Centroids:", vector)
    print("Counts:", all_counts)

    
    print("Contenu brut du Vector:", vector.coefs)
    print("Tableau numpy:\n", vector.coefs["centroids"])
    print("Version liste Python:", vector.coefs["centroids"].tolist())

if __name__ == "__main__":
    test()