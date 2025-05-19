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

"""Messages for the default Federated Unsupervised Learning process of DecLearn."""

import dataclasses
from typing import Any, Dict, List, Optional, Tuple

from typing_extensions import Self  # future: import from typing (py >=3.11)

from declearn.aggregator import Aggregator, ModelUpdates
from declearn.messaging._api import Message
from declearn.metrics import MetricInputType, MetricState
from declearn.model.api import Model, Vector
from declearn.optimizer import Optimizer
from declearn.optimizer.modules import AuxVar
from declearn.utils import deserialize_object, serialize_object


__all__ = [
    "KMeansInitRequest",
    "KMeansInitReply",
    "KMeansdStopTraining",
    "KMeansTrainRequest",
    "KMeansTrainReply",
]

@dataclasses.dataclass
class KMeansInitRequest(Message):
    """Server-emitted request to initialize local kmeans model."""
    typekey = "kmeans_init_request"
    k_global: int
    privacy_threshold: int = 2

@dataclasses.dataclass
class KMeansInitReply(Message):
    """Client-emitted message indicating that initialization went fine."""
    typekey = "kmeans_init_reply"
    cluster_means: List[Vector]
    sample_counts: List[int]


@dataclasses.dataclass
class KMeansdStopTraining(Message):
    """Server-emitted notification that the training process is over."""
    typekey = "stop_training"
    centroids: Vector
    rounds: int


@dataclasses.dataclass
class KMeansTrainRequest(Message):
    """Server-emitted request to participate in a training round."""
    typekey = "kmeans_iter_request"
    centroids: List[Vector]
    round_i: int

@dataclasses.dataclass
class KMeansTrainReply(Message):
    """Client-emitted results from a local training round."""
    typekey = "kmeans_iter_reply"
    cluster_means: List[Vector]
    sample_counts: List[int]

