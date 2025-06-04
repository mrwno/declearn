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
from typing import Union

from declearn import messaging
from declearn.dataset import Dataset
from declearn.model.api import Model
from declearn.utils import get_logger

__all__ = [
    "UnsupervisedTrainingManager",
]


class UnsupervisedTrainingManager:
    """Class wrapping the logic for local training and evaluation rounds."""

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
        message: messaging.KMeansTrainRequest,
    ) -> Union[messaging.KMeansTrainReply, messaging.Error]:
        """Run a local training round.

        If an exception is raised during the local process, wrap it as
        an Error message instead of raising it.

        Parameters
        ----------
        message: KMeansTrainRequest
            Instructions from the server regarding the training round.

        Returns
        -------
        reply: KMeansTrainReply or Error
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
        message: messaging.KMeansTrainRequest,
    ) -> messaging.KMeansTrainReply:
        """Backend to `training_round`, without exception capture hooks."""

        self.logger.info("Applying server updates to local objects.")
        self.model.set_weights(message.centroids)
        result = self.model.compute_kmeans(
            self.train_data,
            None,
            True
        )

        self.model.set_weights(result["centroids"])
        self.logger.info(
            "Training round %s completed",
            message.round_i,
        )
       
        return messaging.KMeansTrainReply(
            cluster_means=result["centroids"],
            sample_counts=result["counts"],
        )
