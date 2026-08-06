import io
import pickle
from typing import Any

import torch


def torch_load_compat(path: str, map_location: torch.device) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _torch_load_from_bytes(data: bytes, map_location: torch.device) -> Any:
    buffer = io.BytesIO(data)
    try:
        return torch.load(buffer, map_location=map_location, weights_only=False)
    except TypeError:
        buffer.seek(0)
        return torch.load(buffer, map_location=map_location)


class TorchDeviceUnpickler(pickle.Unpickler):
    def __init__(self, file, map_location: torch.device):
        super().__init__(file)
        self.map_location = map_location

    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda data: _torch_load_from_bytes(data, self.map_location)
        return super().find_class(module, name)


def pickle_load_torch_device(file, map_location: torch.device) -> Any:
    return TorchDeviceUnpickler(file, map_location).load()
