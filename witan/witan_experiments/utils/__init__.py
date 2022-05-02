from datetime import datetime
from inspect import signature
import random
import numpy as np
import pandas as pd
import torch
from typing import Dict, Sequence, Any, cast


__LOG_FILE__ = None


def new_log_file() -> None:
    """Switch to a new log file, useful to switch log file when starting
    to log from a new thread."""
    global __LOG_FILE__
    __LOG_FILE__ = datetime.now().strftime('logs/%y-%m-%d_%H%M%S.log')


def log(message: str) -> None:
    """Print a log message and write to the log file."""
    logline = f'{message} - {datetime.now()}'
    print(logline)

    if __LOG_FILE__ is None:
        new_log_file()
    with open(cast(str, __LOG_FILE__), 'a') as logfile:
        logfile.write(f'{logline}\n')


def torch_weight_reset(model: torch.nn.Module) -> None:
    """Reset the weights of a torch model."""
    if isinstance(model, torch.nn.Linear):
        model.reset_parameters()


def torch_seed_reset() -> None:
    """Reset random seeds used by torch."""
    # Fixed seeding for Torch (https://pytorch.org/docs/stable/notes/randomness.html)
    torch.manual_seed(1)
    random.seed(2)
    np.random.seed(3)


def prefix_keys(dictionary: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """Prefix a dict with all keys in the given dictionary prefixed."""
    return {'{}{}'.format(prefix, key): value
            for key, value in dictionary.items()}


def inverse_dict(dictionary: Dict[Any, Any]) -> Dict[Any, Any]:
    """Return a dict that inverses the mapping of the given dict -
    i.e. values map to keys."""
    return {value: key for key, value in dictionary.items()}


def drop_keys(dictionary: Dict[Any, Any], keys: Sequence[Any]) -> Dict[Any, Any]:
    """Return a dict that drops any of the given keys from the given dictionary."""
    return {k: v for k, v in dictionary.items() if k not in set(keys)}


def list_series_to_one_hot_df(list_series: pd.Series) -> pd.DataFrame:
    """Convert a Series of lists (allowing nans) into a DataFrame of
    one-hot binary columns representing the presence of each possible
    list item."""
    list_series = list_series.copy()
    list_series[list_series.isna()] = cast(pd.Series, list_series[list_series.isna()]).apply(lambda x: [])
    element_set = set([element for elements in list_series for element in elements])
    return pd.DataFrame({
        element: list_series.apply(lambda elements: (element in elements) if elements else False)
        for element in sorted(element_set)
    })


class KeyHashable:
    """Mixin for objects that can be treated as a configuration object
    that can be hashed to a unique key."""

    def __init__(self):
        pass

    def key(self) -> str:
        """Return the unique key for this object."""
        def format_value(value):
            if isinstance(value, bool):
                return 'T' if value else 'F'
            return value

        param_str = ''.join([
            f'-{param.name}{format_value(getattr(self, param.name))}'
            for param in signature(self.__class__.__init__).parameters.values()
            if param.name not in ['self']
        ])
        return f'{self.__class__.__name__}{param_str}'.replace('_', '')

    def __repr__(self) -> str:
        param_str = ', '.join([
            f'{param.name}={getattr(self, param.name)}'
            for param in signature(self.__class__.__init__).parameters.values()
            if param.name not in ['self']
        ])
        return f'{self.__class__.__name__}({param_str})'

    def __hash__(self) -> int:
        return hash(self.key())

    def __eq__(self, other: Any) -> bool:
        return self.__hash__() == other.__hash__()

    def __lt__(self, other: Any) -> bool:
        return self.__hash__() < other.__hash__()
