# coding: utf-8
# AI generated file
"""Demonstration script using the Blobs dataset."""

import os
import tempfile
from typing import Literal, Optional

import fire # type: ignore
from declearn.test_utils import generate_ssl_certificates, make_importable
from declearn.utils import run_as_processes

with make_importable(os.path.dirname(__file__)):
    from prepare_data import prepare_blobs
    from run_client import run_client
    from run_server import run_server

def run_demo(
    nb_clients: int = 3,
    n_samples: int = 300,
    n_features: int = 2,
    centers: int = 3,
    cluster_std: float = 1.0, 
    scheme: Literal["iid", "clusters"] = "iid",
    seed: Optional[int] = 42,
) -> None:
    """Run a server and its clients using multiprocessing with blob data."""
    # Generate blob data
    data_folder = prepare_blobs(
        nb_clients, 
        n_samples, 
        n_features, 
        centers, 
        cluster_std,  
        scheme, 
        seed=seed
    )
    
    with tempfile.TemporaryDirectory() as folder:
        ca_cert, sv_cert, sv_pkey = generate_ssl_certificates(folder)
        
        server = (run_server, (nb_clients,), {
            "certificate": sv_cert, 
            "private_key": sv_pkey
        })
        
        client_kwargs = {
            "data_folder": data_folder, "ca_cert": ca_cert, "verbose": False
        }
        clients = [
            (run_client, (f"client_{idx}",), client_kwargs)
            for idx in range(nb_clients)
        ]
        
        success, outp = run_as_processes(server, *clients)
        if not success:
            raise RuntimeError(
                "Something went wrong during the demo. Exceptions caught:\n"
                "\n".join(str(e) for e in outp if isinstance(e, RuntimeError))
            )

if __name__ == "__main__":
    fire.Fire(run_demo)