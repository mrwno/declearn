# coding: utf-8

import os
import datetime
import logging
import fire
import numpy as np  # type: ignore

import declearn
from declearn.main import UnsupervisedFederatedClient
from declearn.dataset import InMemoryDataset

FILEDIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CERT = os.path.join(FILEDIR, "ca-cert.pem")

def run_client(
    client_name: str,
    data_folder: str,
    ca_cert: str = DEFAULT_CERT,
    protocol: str = "websockets",
    serv_uri: str = "wss://localhost:8765",
    verbose: bool = True,
) -> None:
    """Instantiate and run a given client for federated K-means."""
    # Set CPU as device
    declearn.utils.set_device_policy(gpu=False)

    # Set up logger
    stamp = datetime.datetime.now().strftime("%y-%m-%d_%H-%M")
    checkpoint = os.path.join(FILEDIR, f"result_{stamp}", client_name)
    logger = declearn.utils.get_logger(
        name=client_name,
        fpath=os.path.join(checkpoint, "logs.txt"),
    )

    # Reduce logger verbosity
    if not verbose:
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setLevel(declearn.utils.LOGGING_LEVEL_MAJOR)

    # Load data
    data_folder = os.path.join(FILEDIR, data_folder, client_name)
    X = np.load(os.path.join(data_folder, "train_data.npy"))
    # K-means doesn't use labels, create dummy ones
    y = np.zeros(len(X))
    dataset = InMemoryDataset(X, y)

    # Network configuration
    network = declearn.communication.build_client(
        protocol=protocol,
        server_uri=serv_uri,
        name=client_name,
        certificate=ca_cert,
    )

    # Instantiate and run unsupervised client
    client = UnsupervisedFederatedClient(
        netwk=network,
        train_data=dataset,
        logger=logger,
        verbose=verbose,
    )
    client.run()

def main():
    fire.Fire(run_client)

if __name__ == "__main__":
    main()