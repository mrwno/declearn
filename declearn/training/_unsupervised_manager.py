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

"""Wrapper to run local training and evaluation rounds in a unsupervised FL process."""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import time

import numpy as np
import tqdm

from declearn import messaging
from declearn.aggregator import Aggregator
from declearn.dataset import Dataset
from declearn.metrics import (
    MeanMetric,
    Metric,
    MetricInputType,
    MetricSet,
    MetricState,
)
from declearn.model.api import Model
from declearn.optimizer import Optimizer
from declearn.training._constraints import (
    Constraint,
    ConstraintSet,
    TimeoutConstraint,
)
from declearn.typing import Batch
from declearn.utils import LOGGING_LEVEL_MAJOR, get_logger

__all__ = [
    "UnsupervisedTrainingManager",
]


class UnsupervisedTrainingManager:
    """Class wrapping the logic for local training and evaluation rounds."""

    # one too-many attribute; pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        model: Model,
        train_data: Dataset,
        logger: Union[logging.Logger, str, None] = None,
        verbose: bool = True,
    ) -> None:
        """Instantiate the client-side training and evaluation process.

        Parameters
        ----------
        model: Model
            Model instance that needs training and/or evaluating.
        train_data: Dataset
            Dataset instance wrapping the local training dataset.
        logger: logging.Logger or str or None, default=None,
            Logger to use, or name of a logger to set up with
            `declearn.utils.get_logger`.
            If None, use `type(self).__name__`.
        verbose: bool, default=True
            Whether to display progress bars when running training
            and validation rounds.
        """
        # arguments serve modularity; pylint: disable=too-many-arguments
        self.model = model
        self.train_data = train_data
        if not isinstance(logger, logging.Logger):
            logger = get_logger(logger or f"{type(self).__name__}")
        self.logger = logger
        self.verbose = verbose

    def training_round(
        self,
        message: messaging.TrainRequest,
    ) -> Union[messaging.TrainReply, messaging.Error]:
        """Run a local training round.

        If an exception is raised during the local process, wrap it as
        an Error message instead of raising it.

        Parameters
        ----------
        message: TrainRequest
            Instructions from the server regarding the training round.

        Returns
        -------
        reply: TrainReply or Error
            Message wrapping results from the training round, or any
            error raised during it.
        """
        self.logger.info("Participating in training round %s", message.round_i)
        # Try running the training round; return the reply is successful.
        try:
            return self._training_round(message)
        # In case of failure, wrap the exception as an Error message.
        except Exception as exception:  # pylint: disable=broad-except
            self.logger.error(
                "Error encountered during training: %s.", exception
            )
            return messaging.Error(repr(exception))

    def _training_round(
        self,
        message: messaging.TrainRequest,
    ) -> messaging.TrainReply:
        """Backend to `training_round`, without exception capture hooks."""
        # Unpack and apply model weights and optimizer auxiliary variables.

        start_time = time.time()

        self.logger.info("Applying server updates to local objects.")
        # Line 13 of the FKM paper
        if message.weights is not None:
           self.model.set_weights(message.weights, trainable=True) #remplace les centroides du modele par ceux du message
        
        # Line 14 of the FKM paper
        updates = self.model.local_kmeans_iteration(self.train_data)

        # Train under instructed effort constraints.
        params = message.n_epoch, message.n_steps, message.timeout
        self.logger.info(
            "Training local model for %s epochs | %s steps | %s seconds.",
            *params,
        )
        
        t_spent = time.time() - start_time
        # Wrap them as a TrainReply together with effort metadata and return.
        return messaging.TrainReply(
            updates=updates,
            aux_var={},
            n_epoch=1,
            n_steps=1,
            t_spent=round(t_spent, 3),
        )
