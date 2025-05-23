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

"""Client-side main Unsupervised Federated Learning orchestrating class."""

import asyncio
import dataclasses
import logging
import os
import warnings
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from declearn import messaging
from declearn.communication.api import NetworkClient
from declearn.communication.utils import (
    NetworkClientConfig,
    verify_server_message_validity,
)
from declearn.dataset import Dataset
from declearn.messaging import Message, SerializedMessage
from declearn.training import UnsupervisedTrainingManager
from declearn.utils import LOGGING_LEVEL_MAJOR, get_logger


__all__ = [
    "UnsupervisedFederatedClient",
]


class UnsupervisedFederatedClient:
    """Client-side Unsupervised Federated Learning orchestrating class."""

    # one-too-many attribute; pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        netwk: Union[NetworkClient, NetworkClientConfig, Dict[str, Any], str],
        train_data: Union[Dataset, str],
        logger: Union[logging.Logger, str, None] = None,
        verbose: bool = True,
    ) -> None:
        """Instantiate a client to participate in a federated learning task.

        Parameters
        ----------
        netwk: NetworkClient or NetworkClientConfig or dict or str
            NetworkClient communication endpoint instance, or configuration
            dict, dataclass or path to a TOML file enabling its instantiation.
            In the latter three cases, the object's default logger will be set
            to that of this `FederatedClient`.
        train_data: Dataset or str
            Dataset instance wrapping the training data.
            (DEPRECATED) May be a path to a JSON dump file.
        logger: logging.Logger or str or None, default=None,
            Logger to use, or name of a logger to set up with
            `declearn.utils.get_logger`.
            If None, use `type(self):netwk.name`.
        verbose: bool, default=True
            Whether to verbose about ongoing operations.
            If True, display progress bars during training and validation
            rounds. If False and `logger is None`, set the logger's level
            to filter off most routine information.
        """
        # arguments serve modularity; pylint: disable=too-many-arguments
        # Assign the wrapped NetworkClient.
        self.netwk, replace_netwk_logger = self._parse_netwk(netwk)
        # Assign the logger and optionally replace that of the network client.
        if not isinstance(logger, logging.Logger):
            logger = get_logger(
                name=logger or f"{type(self).__name__}-{self.netwk.name}",
                level=logging.INFO if verbose else LOGGING_LEVEL_MAJOR,
            )
        self.logger = logger
        if replace_netwk_logger:
            self.netwk.logger = self.logger
        # Assign the wrapped training dataset.
        if not isinstance(train_data, Dataset):
            raise TypeError("'train_data' should be a Dataset.")
        self.train_data = train_data
        self.verbose = bool(verbose)
        self.model = None
        # Create slots that are (opt.) populated during initialization.
        # Using trainmanager later, now we just use the model directly
        self.trainmanager = None  # type: Optional[UnsupervisedTrainingManager]

    @staticmethod
    def _parse_netwk(netwk) -> Tuple[NetworkClient, bool]:
        """Parse 'netwrk' instantiation argument.

        Return both a 'NetworkClient' instance and a bool indicating
        whether that instance's logger should be replaced with that
        of the client (set up at a latter step).
        """
        # Case when a NetworkClient instance is provided: return.
        if isinstance(netwk, NetworkClient):
            return netwk, False
        # Case when a NetworkClientConfig is expected: verify or parse.
        if isinstance(netwk, NetworkClientConfig):
            config = netwk
        elif isinstance(netwk, str):
            config = NetworkClientConfig.from_toml(netwk)
        elif isinstance(netwk, dict):
            replace_netwk_logger = netwk.get("logger", None) is None
            config = NetworkClientConfig.from_params(**netwk)
        else:
            raise TypeError(
                "'netwk' should be a 'NetworkClient' instance or the valid "
                f"configuration of one, not '{type(netwk)}'"
            )
        # Instantiate from the (parsed) config.
        replace_netwk_logger = config.logger is None
        return config.build_client(), replace_netwk_logger

    def run(
        self,
    ) -> None:
        """Participate in the federated learning process.

        * Connect to the orchestrating `FederatedServer` and register
          for training, sharing some metadata about `self.train_data`.
        * Await initialization instructions to spawn the Model that is
          to be trained and the local Optimizer used to do so.
        * Participate in training and evaluation rounds based on the
          server's requests, checkpointing the model and local loss.
        * Expect instructions to stop training, or to cancel it in
          case errors are reported during the process.
        """
        asyncio.run(self.async_run())

    async def async_run(
        self,
    ) -> None:
        """Participate in the federated learning process.

        Note: this method is the async backend of `self.run`.
        """
        async with self.netwk:
            # Register for training, then collect initialization information.
            await self.register()
            await self.initialize()
            # Process server instructions as they come.
            while True:
                message = await self.netwk.recv_message()
                stoprun = await self.handle_message(message)
                if stoprun:
                    break

    async def handle_message(
        self,
        message: SerializedMessage,
    ) -> bool:
        """Handle an incoming message from the server.

        Parameters
        ----------
        message: SerializedMessage
            Serialized message that needs triage and processing.

        Returns
        -------
        exit_loop: bool
            Whether to interrupt the client's message-receiving loop.
        """
        exit_loop = False
        if issubclass(message.message_cls, messaging.KMeansTrainRequest):
            await self.training_round(message.deserialize())
        elif issubclass(message.message_cls, messaging.KMeansdStopTraining):
            await self.stop_training(message.deserialize())
            exit_loop = True
        elif issubclass(message.message_cls, messaging.CancelTraining):
            await self.cancel_training(message.deserialize())
        else:
            error = "Unexpected message type received from server: "
            error += message.message_cls.__name__
            self.logger.error(error)
            raise ValueError(error)
        return exit_loop

    async def register(
        self,
    ) -> None:
        """Register for participation in the federated learning process.

        Raises
        ------
        RuntimeError
            If registration has failed 10 times (with a 1 minute delay
            between connection and registration attempts).
        """
        for i in range(10):  # max_attempts (10)
            self.logger.info(
                "Attempting to join training (attempt n°%s)", i + 1
            )
            registered = await self.netwk.register()
            if registered:
                break
            await asyncio.sleep(60)  # delay_retries (1 minute)
        else:
            raise RuntimeError("Failed to register for training.")

    async def initialize(
        self,
    ) -> None:
        """Set up a Model and an Optimizer based on server instructions.

        Await server instructions (as an InitRequest message) and conduct
        initialization.

        Raises
        ------
        RuntimeError
            If initialization failed, either because the message was not
            received or was of incorrect type, or because instantiation
            of the objects it specifies failed.

        Returns
        -------
        model: Model
            Model that is to be trained (with shared initial parameters).
        optim: Optimizer
            Optimizer that is to be used locally to train the model.
        """
        # Await initialization instructions.
        self.logger.info("Awaiting initialization instructions from server.")
        received = await self.netwk.recv_message()
        # If a MetadataQuery is received, process it, then await InitRequest.
        if issubclass(received.message_cls, messaging.MetadataQuery):
            await self._collect_and_send_metadata(received.deserialize())
            received = await self.netwk.recv_message()
        # Ensure that an 'InitRequest' was received.
        message = await verify_server_message_validity(
            self.netwk, received, expected=messaging.KMeansInitRequest
        )

        self.model = message.model


        #choose k data point as centroids and do 1 iteration of kmeans (instead of kmeans++)
        # Create KMeansInitReply message to send back to the server.
        
        # Send back an empty message to indicate that things went fine.
        self.logger.info("Notifying the server that initialization went fine.")

        # ICI on choisit des centroïdes initiaux aléatoires et on les envoie au serveur
        
        await self.netwk.send_message(messaging.InitReply())

    async def _collect_and_send_metadata(
        self,
        message: messaging.MetadataQuery,
    ) -> None:
        """Collect and report some metadata based on server instructions."""
        self.logger.info("Collecting metadata to send to the server.")
        metadata = dataclasses.asdict(self.train_data.get_data_specs())
        if missing := set(message.fields).difference(metadata):
            err_msg = f"Metadata query for undefined fields: {missing}."
            await self.netwk.send_message(messaging.Error(err_msg))
            raise RuntimeError(err_msg)
        data_info = {key: metadata[key] for key in message.fields}
        self.logger.info(
            "Sending training dataset metadata to the server: %s.",
            list(data_info),
        )
        await self.netwk.send_message(messaging.MetadataReply(data_info))


    async def training_round(
        self,
        message: messaging.KMeansTrainRequest,
    ) -> None:
        """Run a local training round.

        If an exception is raised during the local process, wrap
        it as an Error message and send it to the server instead
        of raising it.

        Parameters
        ----------
        message: KMeansTrainRequest
            Instructions from the server regarding the training round.
        """
        # Update centroids with those from the server.

        # Run a local training round.

        # Create a KMeansTrainReply message to send back to the server.
        await self.netwk.send_message(reply)


    async def stop_training(
        self,
        message: messaging.KMeansdStopTraining,
    ) -> None:
        """Handle a server request to stop training.

        Parameters
        ----------
        message: KMeansdStopTraining
            KMeansdStopTraining message received from the server.
        """
        self.logger.info(
            "Training is now over, after %s rounds.",
            message.rounds,
        )

        # Update the model with the final centroids.
        

    async def cancel_training(
        self,
        message: messaging.CancelTraining,
    ) -> None:
        """Handle a server request to cancel training.

        Parameters
        ----------
        message: CancelTraining
            CancelTraining message received from the server.
        """
        error = "Training was cancelled by the server, with reason:\n"
        error += message.reason
        self.logger.warning(error)
        raise RuntimeError(error)
