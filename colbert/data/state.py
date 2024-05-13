from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer

from colbert.data.dataset import MultiDataset


@dataclass
class State:
    """A data container representing information on the current run."""

    epoch: int = 0
    batch: int = 0
    train_step: int = 0

    loss: float = 0.0
    mean_similarity: float = 0.0

    metrics: List[Tuple[Dict[str, Any], int]] = field(default_factory=lambda: [])

    model: nn.Module = None
    optimizer: Optimizer = None
    dataset: Optional[MultiDataset] = None

    def state_dict(self) -> Dict[str, Any]:
        """State dict, used for saving the object for checkpointing."""
        ret = {key: value for key, value in self.__dict__.items()}
        if ret['dataset'] is not None:
            # del ret['dataset']
            ret['dataset'] = ret['dataset'].state_dict()
        if isinstance(ret['loss'], Tensor):
            ret['loss'] = ret['loss'].item()
        if isinstance(ret['mean_similarity'], Tensor):
            ret['mean_similarity'] = ret['mean_similarity'].item()
        return ret

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Loads the state.

        Args:
            state_dict (dict): The state. Should be an object returned
                from a call to `state_dict`.
        """
        self.__dict__.update(state_dict)
