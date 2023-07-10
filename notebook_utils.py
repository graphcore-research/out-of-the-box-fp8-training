# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import inspect
from typing import Any, Dict

from torch import nn

from nanoGPT.config import train_shakespeare_char
from nanoGPT.model import GPTConfig


def config_dict_from_module(module) -> Dict[str, Any]:
    return {k: v for k, v in vars(module).items() if not k.startswith("__")}


def extract_model_params(config):
    return {
        k: v
        for k, v in config.items()
        if k in inspect.signature(GPTConfig).parameters.keys()
    }


_general_config_dict = config_dict_from_module(train_shakespeare_char)
_model_config_dict = extract_model_params(_general_config_dict)
_model_config_dict["vocab_size"] = 65  # Generated from data/shakespeare_char/prepare.py
config = GPTConfig(**_model_config_dict)


def train(model: nn.Module) -> None:
    print("Not implemented yet...", model)
