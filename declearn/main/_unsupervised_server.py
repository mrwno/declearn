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

"""Server-side main Unsupervised Federated Learning orchestrating class."""

import asyncio
import logging
from typing import (
    # fmt: off
    Any, Dict, Set, Type, TypeVar, Union
)
import numpy as np

from declearn import messaging
from declearn.communication import NetworkServerConfig
from declearn.communication.api import NetworkServer
from declearn.main.config import (
    FLRunConfig,
)
from declearn.main.utils import (
    AggregationError,
    aggregate_clients_data_info,
)
from declearn.model.api import Model, Vector
from declearn.utils import get_logger
from declearn.model.kmeans import FederatedKMeansModel


__all__ = [
    "UnsupervisedFederatedServer",
]


MessageT = TypeVar("MessageT", bound=messaging.Message)


class UnsupervisedFederatedServer:
    """Server-side UnsupervisedFederated Learning orchestrating class."""

    # one-too-many attribute; pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        model: Model,
        netwk: Union[NetworkServer, NetworkServerConfig, Dict[str, Any], str],
        logger: Union[logging.Logger, str, None] = None,
    ) -> None:
        """Instantiate the orchestrating server for a federated learning task.

        Parameters
        ----------
        model: Model or dict or str
            Model instance, that may be serialized as an ObjectConfig,
            a config dict or a JSON file the path to which is provided.
        netwk: NetworkServer or NetworkServerConfig or dict or str
            NetworkServer communication endpoint instance, or configuration
            dict, dataclass or path to a TOML file enabling its instantiation.
            In the latter three cases, the object's default logger will
            be set to that of this `FederatedServer`.
        logger: logging.Logger or str or None, default=None,
            Logger to use, or name of a logger to set up with
            `declearn.utils.get_logger`. If None, use `type(self)`.
        """
        # arguments serve modularity; pylint: disable=too-many-arguments
        # Assign the logger.
        if not isinstance(logger, logging.Logger):
            logger = get_logger(logger or type(self).__name__)
        self.logger = logger
        # Assign the wrapped Model.
        self.model = self._parse_model(model) # type: FederatedKMeansModel
        # Assign the wrapped NetworkServer.
        self.netwk = self._parse_netwk(netwk, logger=self.logger)

        self.all_centroids = [] 
        self.all_counts = []
        self.privacy_threshold = 2 
        


    @staticmethod
    def _parse_model(
        model: Model,
    ) -> Model:
        """Parse 'model' instantiation argument."""
        if isinstance(model, Model):
            return model
        raise TypeError(
            "'model' should be a declearn Model, optionally in serialized "
            f"form, not '{type(model)}'"
        )

    @staticmethod
    def _parse_netwk(
        netwk: Union[NetworkServer, NetworkServerConfig, Dict[str, Any], str],
        logger: logging.Logger,
    ) -> NetworkServer:
        """Parse 'netwk' instantiation argument."""
        # Case when a NetworkServer instance is provided: return.
        if isinstance(netwk, NetworkServer):
            return netwk
        # Case when a NetworkServerConfig is expected: verify or parse.
        if isinstance(netwk, NetworkServerConfig):
            config = netwk
        elif isinstance(netwk, str):
            config = NetworkServerConfig.from_toml(netwk)
        elif isinstance(netwk, dict):
            config = NetworkServerConfig(**netwk)
        else:
            raise TypeError(
                "'netwk' should be a 'NetworkServer' instance or the valid "
                f"configuration of one, not '{type(netwk)}'."
            )
        # Instantiate from the (parsed) config.
        if config.logger is None:
            config.logger = logger
        return config.build_server()

    def run(
        self,
        config: Union[FLRunConfig, str, Dict[str, Any]],
    ) -> None:
        """Orchestrate the federated learning routine.

        Parameters
        ----------
        config: FLRunConfig or str or dict
            Container instance wrapping grouped hyper-parameters that
            specify the federated learning process, including clients
            registration, training and validation rounds' setup, plus
            an optional early-stopping criterion.
            May be a str pointing to a TOML configuration file.
            May be as a dict of keyword arguments to be parsed.
        """
        if isinstance(config, dict):
            config = FLRunConfig.from_params(**config)
        if isinstance(config, str):
            config = FLRunConfig.from_toml(config)
        if not isinstance(config, FLRunConfig):
            raise TypeError("'config' should be a FLRunConfig object or str.")
        asyncio.run(self.async_run(config))

    async def async_run(
        self,
        config: FLRunConfig,
    ) -> None:
        """Orchestrate the federated learning routine.

        Note: this method is the async backend of `self.run`.

        Parameters
        ----------
        config: FLRunConfig
            Container instance wrapping grouped hyper-parameters that
            specify the federated learning process, including clients
            registration, training and validation rounds' setup, plus
            optional elements: local differential-privacy parameters,
            fairness evaluation parameters, and/or an early-stopping
            criterion.
        """
        # Start the communications server and run the FL process.
        async with self.netwk:
            # Conduct the initialization phase.
            await self.initialization(config)
            # Iteratively run training and evaluation rounds.
            round_i = 0
            for round_i in range(config.rounds):
                await self.training_round(round_i)
            # Interrupt training when time comes.
            self.logger.info("Stopping training.")
            await self.stop_training(round_i)

    async def initialization(
        self,
        config: FLRunConfig,
    ) -> None:
        """Orchestrate the initialization steps to set up training.

        Wait for clients to register and process their data information.
        Send instructions to clients to set up their model and optimizer.
        Await clients to have finalized their initialization step; raise
        and cancel training if issues are reported back.

        Parameters
        ----------
        config: FLRunConfig
            Container instance wrapping hyper-parameters that specify
            the planned federated learning process, including clients
            registration ones as a RegisterConfig dataclass instance.

        Raises
        ------
        RuntimeError
            In case any of the clients returned an Error message rather
            than an Empty ping-back message. Send CancelTraining to all
            clients before raising.
        """
        # Gather the RegisterConfig instance from the main FLRunConfig.
        regst_cfg = config.register
        # Wait for clients to register.
        self.logger.info("Starting clients registration process.")
        await self.netwk.wait_for_clients(
            regst_cfg.min_clients, regst_cfg.max_clients, regst_cfg.timeout
        )
        self.logger.info("Clients' registration is now complete.")
        # When needed, prompt clients for metadata and process them.
        await self._require_and_process_data_info()
        # Serialize intialization information and send it to clients.
        message = messaging.KMeansInitRequest(
            k_global=self.model.n_clusters,
            privacy_threshold=self.privacy_threshold
        )
        self.logger.info("Sending initialization requests to clients.")
        await self.netwk.broadcast_message(message)
        # Await a confirmation from clients that initialization went well.
        # If any client has failed to initialize, raise.
        self.logger.info("Waiting for clients' responses.")
        replies = await self._collect_results(
            clients=self.netwk.client_names,
            msgtype=messaging.KMeansInitReply,
            context="Initialization"
        )
        # Concatenate centroids and counts from all clients.
        all_centroids = []
        all_counts = []
        for reply in replies.values():
            centroids = reply.cluster_means.coefs["centroids"]
            all_centroids.extend(centroids)
            all_counts.extend(reply.sample_counts)
        #all_centroids_vector = Vector.build({"centroids" :np.array(all_centroids)})
        self.all_centroids = all_centroids
        self.all_counts = all_counts
        
        self.logger.info("Initialization was successful.")

    async def _require_and_process_data_info(
        self,
    ) -> None:
        """Collect, validate, aggregate and make use of clients' data-info.

        Raises
        ------
        AggregationError
            In case (some of) the clients' data info is invalid, or
            incompatible. Send CancelTraining to all clients before
            raising.
        """
        fields = self.model.required_data_info  # revise: add optimizer, etc.
        if not fields:
            return
        # Collect required metadata from clients.
        query = messaging.MetadataQuery(list(fields))
        await self.netwk.broadcast_message(query)
        replies = await self._collect_results(
            self.netwk.client_names,
            msgtype=messaging.MetadataReply,
            context="Metadata collection",
        )
        clients_data_info = {
            client: reply.data_info for client, reply in replies.items()
        }
        # Try aggregating the input data_info.
        try:
            info = aggregate_clients_data_info(clients_data_info, fields)
        # In case of failure, cancel training, notify clients, log and raise.
        except AggregationError as exc:
            messages = {
                client: messaging.CancelTraining(reason)
                for client, reason in exc.messages.items()
            }
            await self.netwk.send_messages(messages)
            self.logger.error(exc.error)
            raise exc
        # Otherwise, initialize the model based on the aggregated information.
        self.model.initialize(info)

    async def _collect_results(
        self,
        clients: Set[str],
        msgtype: Type[MessageT],
        context: str,
    ) -> Dict[str, MessageT]:
        """Collect some results sent by clients and ensure they are okay.

        Parameters
        ----------
        clients: set[str]
            Names of the clients that are expected to send messages.
        msgtype: type[messaging.Message]
            Type of message that clients are expected to send.
        context: str
            Context of the results collection (e.g. "training" or
            "evaluation"), used in logging or error messages.

        Raises
        ------
        RuntimeError
            If any client sent an incorrect message or reported
            failure to conduct the evaluation step properly.
            Send CancelTraining to all clients before raising.

        Returns
        -------
        results: dict[str, `msgtype`]
            Client-wise collected messages.
        """
        # Await clients' responses and type-check them.
        replies = await self.netwk.wait_for_messages(clients)
        results = {}  # type: Dict[str, MessageT]
        errors = {}  # type: Dict[str, str]
        for client, reply in replies.items():
            if issubclass(reply.message_cls, msgtype):
                results[client] = reply.deserialize()
            elif issubclass(reply.message_cls, messaging.Error):
                err_msg = reply.deserialize().message
                errors[client] = f"{context} failed: {err_msg}"
            else:
                errors[client] = f"Unexpected message: {reply.message_cls}"
        # If any client has failed to send proper results, raise.
        # future: modularize errors-handling behaviour
        if errors:
            err_msg = f"{context} failed for another client."
            messages = {
                client: messaging.CancelTraining(errors.get(client, err_msg))
                for client in self.netwk.client_names
            }  # type: Dict[str, messaging.Message]
            await self.netwk.send_messages(messages)
            err_msg = f"{context} failed for {len(errors)} clients:" + "".join(
                f"\n    {client}: {error}" for client, error in errors.items()
            )
            self.logger.error(err_msg)
            raise RuntimeError(err_msg)
        # Otherwise, return collected results.
        return results

    async def training_round(
        self,
        round_i: int,
    ) -> None:
        """Orchestrate a training round.

        Parameters
        ----------
        round_i: int
            Index of the training round.
        """
        # Select participating clients. Run SecAgg setup when needed.
        self.logger.info("Initiating training round %s", round_i)
        clients = self._select_training_round_participants()
        # Send training instructions and await results.
        await self._send_training_instructions(clients, round_i) # envoie les C_g au clients
        self.logger.info("Awaiting clients' training results.")
        # Reiceive results from clients and check for errors.
        results = await self._collect_results(
            clients, messaging.KMeansTrainReply, "training" #recoit les nouveaux centroids ( S_i et C_i)
        )
        # Aggregate client-wise results and update the global model.
        self.logger.info("Conducting server-side optimization.")
        self._conduct_global_update(results) # Mettre à jour les centroids

    def _select_training_round_participants(
        self,
    ) -> Set[str]:
        """Return the names of clients that should participate in the round."""
        return self.netwk.client_names

    async def _send_training_instructions(
        self,
        clients: Set[str],
        round_i: int,
    ) -> None:
        """Send training instructions to selected clients.

        Parameters
        ----------
        clients: set[str]
            Names of the clients participating in the training round.
        round_i: int
            Index of the training round.
        """
        result = self.model.compute_kmeans(self.all_centroids, weights = self.all_counts, client = False)

        # Create message that contains centroids.
        msg_light = messaging.KMeansTrainRequest( 
            centroids=Vector.build({"centroids": result["centroids"]}), 
            round_i=round_i,
        )
        # Send it to clients, sparingly joining model weights.
        await self.netwk.broadcast_message(msg_light, clients)

    def _conduct_global_update(
        self,
        results: Dict[str, messaging.KMeansTrainReply],
    ) -> None:
        """Use training results from clients to update the global model.

        Parameters
        ----------
        results: dict[str, KMeansTrainReply]
            Client-wise KMeansTrainReply message sent after a training round.
        """
        all_centroids = []
        all_counts = []
        for reply in results.values():
            centroids = reply.cluster_means.coefs["centroids"]
            all_centroids.extend(centroids)
            all_counts.extend(reply.sample_counts)
        self.all_centroids = all_centroids
        self.all_counts = all_counts
        

    async def stop_training(
        self,
        rounds: int,
    ) -> None:
        """Notify clients that training is over and send final information.

        Parameters
        ----------
        rounds: int
            Number of training rounds taken until now.
        """
        self.logger.info("Recovering weights that yielded the lowest loss.")
        message = messaging.KMeansStopTraining(
            centroids=self.model.get_weights(),
            rounds=rounds,
        )
        self.logger.info("Notifying clients that training is over.")
        await self.netwk.broadcast_message(message)