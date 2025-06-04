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

"""Script to run a federated server for unsupervised clustering on the blobs example."""

import datetime
import os

import fire # type: ignore

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
    """Instantiate and run the orchestrating server for federated K-means.

    Arguments
    ---------
    nb_clients: int
        Exact number of clients used in this example.
    certificate: str
        Path to the (self-signed) SSL certificate to use.
    private_key: str
        Path to the associated private-key to use.
    protocol: str, default="websockets"
        Name of the communication protocol to use.
    host: str, default="localhost"
        Hostname or IP address on which to serve.
    port: int, default=8765
        Communication port on which to serve.
    """

    # Set CPU as device
    declearn.utils.set_device_policy(gpu=False)

    # Set up checkpointing and logging
    stamp = datetime.datetime.now().strftime("%y-%m-%d_%H-%M")
    checkpoint = os.path.join(FILEDIR, f"result_{stamp}", "server")
    # Set up a logger, records from which will go to a file.
    logger = declearn.utils.get_logger(
        name="Server",
        fpath=os.path.join(checkpoint, "logs.txt"),
    )

    # Create K-means model
    model = FederatedKMeansModel(
        n_clusters=n_clusters,
        privacy_threshold=2
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

    #Not used, but required for the server to run
    training = declearn.main.config.TrainingConfig(
        batch_size=32,
        n_epoch=1,
    )
    evaluate = declearn.main.config.EvaluateConfig(
        batch_size=128,
    )

    # FL configuration
    register = RegisterConfig(
        min_clients=nb_clients,
        max_clients=nb_clients,
        timeout=300,
    )
    run_config = FLRunConfig.from_params(
        rounds=10,
        register=register,
        training=training,
        evaluate=evaluate,
        privacy=None,
        early_stop=None,
    )
    server.run(run_config)

def main():
    fire.Fire(run_server)

if __name__ == "__main__":
    main()