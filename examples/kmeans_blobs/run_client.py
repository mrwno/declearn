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

"""Script to run a federated client for unsupervised clustering on the blobs example."""


import datetime
import os
import logging

import fire # type: ignore

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
    """Instantiate and run a given client for federated K-means.
    
    Parameters
    ---------
    client_name: str
        Name of the client (i.e. center data from which to use).
    data_folder: str
        The parent folder of this client's data
    ca_cert: str, default="./ca-cert.pem"
        Path to the certificate authority file that was used to
        sign the server's SSL certificate.
    protocol: str, default="websockets"
        Name of the communication protocol to use.
    serv_uri: str, default="wss://localhost:8765"
        URI of the server to which to connect.
    verbose: bool, default=True
        Whether to log everything to the console, or filter out most non-error
        information.
    """
    # Set CPU as device
    declearn.utils.set_device_policy(gpu=False)

    # Set up logger and checkpointer
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

    ### (1-2) Interface training and optional validation data.

    # Target the proper dataset (specific to our blobs setup).
    data_folder = os.path.join(FILEDIR, data_folder, client_name)
    train = InMemoryDataset(
        os.path.join(data_folder, "train_data.npy"),
        os.path.join(data_folder, "train_target.npy"),
    )
    valid = InMemoryDataset(
        os.path.join(data_folder, "valid_data.npy"),
        os.path.join(data_folder, "valid_target.npy"),
    )

    network = declearn.communication.build_client(
        protocol=protocol,
        server_uri=serv_uri,
        name=client_name,
        certificate=ca_cert,
    )

    client = UnsupervisedFederatedClient(
        netwk=network,
        train_data=train,
        logger=logger,
        verbose=verbose,
    )
    client.run()

def main():
    fire.Fire(run_client)

if __name__ == "__main__":
    main()