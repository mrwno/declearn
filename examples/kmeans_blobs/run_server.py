# coding: utf-8

import os
import datetime
import fire

import declearn
from declearn.main import UnsupervisedFederatedServer
from declearn.model.kmeans import FederatedKMeansModel
from declearn.main.config import FLRunConfig, RegisterConfig

FILEDIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CERT = os.path.join(FILEDIR, "server-cert.pem")
DEFAULT_PKEY = os.path.join(FILEDIR, "server-pkey.pem")

def run_server(
    nb_clients: int,
    n_clusters: int = 3,
    certificate: str = DEFAULT_CERT,
    private_key: str = DEFAULT_PKEY,
    protocol: str = "websockets",
    host: str = "localhost",
    port: int = 8765,
) -> None:
    """Instantiate and run the orchestrating server for federated K-means."""
    # Set CPU as device
    declearn.utils.set_device_policy(gpu=False)

    # Set up checkpointing and logging
    stamp = datetime.datetime.now().strftime("%y-%m-%d_%H-%M")
    checkpoint = os.path.join(FILEDIR, f"result_{stamp}", "server")
    logger = declearn.utils.get_logger(
        name="Server",
        fpath=os.path.join(checkpoint, "logs.txt"),
    )

    # Create K-means model
    model = FederatedKMeansModel(
        n_clusters=n_clusters,
        privacy_threshold=5  # Minimum samples per cluster
    )

    # Network configuration
    network = declearn.communication.build_server(
        protocol=protocol,
        host=host,
        port=port,
        certificate=certificate,
        private_key=private_key,
    )

    # Instantiate unsupervised server
    server = UnsupervisedFederatedServer(
        model=model,
        netwk=network,
        logger=logger,
    )
    #Not used
    training = declearn.main.config.TrainingConfig(
        batch_size=32,
        n_epoch=1,
    )

    # FL configuration
    register = RegisterConfig(
        min_clients=nb_clients,
        max_clients=nb_clients,
        timeout=300,
    )
    run_config = FLRunConfig.from_params(
        rounds=10,  # Number of federated rounds
        register=register,
        training=training,
    )
    server.run(run_config)

def main():
    fire.Fire(run_server)

if __name__ == "__main__":
    main()