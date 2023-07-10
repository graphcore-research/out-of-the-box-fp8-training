# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import inspect
from typing import Any, Dict

import matplotlib.axes
import pandas as pd
import seaborn as sns
from torch import nn

from nanoGPT.config import train_shakespeare_char
from nanoGPT.model import GPTConfig
from train import run_training


def config_dict_from_module(module) -> Dict[str, Any]:
    return {k: v for k, v in vars(module).items() if not k.startswith("__")}


def extract_model_params(config):
    return {
        k: v
        for k, v in config.items()
        if k in inspect.signature(GPTConfig).parameters.keys()
    }


_general_config_dict = config_dict_from_module(train_shakespeare_char)
_general_config_dict["compile"] = False  # We'll do this in the notebook when necessary
_model_config_dict = extract_model_params(_general_config_dict)
_model_config_dict["vocab_size"] = 65  # Generated from data/shakespeare_char/prepare.py
config = GPTConfig(**_model_config_dict)


def _gen_experiment_name(model: nn.Module) -> str:
    name = model.__class__.__name__.lower()
    backend_names = [b.__qualname__ for b in getattr(model, "backends", [])]
    print(backend_names)
    if backend_names:
        us = any("unit_scaling" in b for b in backend_names)
        fp8 = any("quantisation" in b for b in backend_names)
        if us and not fp8:
            return f"unit_scaled_{name}"
        if fp8 and not us:
            return f"fp8_{name}"
        if us and fp8:
            return f"unit_scaled_fp8_{name}"
    return name


def plot(df: pd.DataFrame) -> matplotlib.axes.Axes:
    sns.set_theme()
    p = sns.lineplot(data=df, x="Steps", y="Loss", hue="Format", style="Train/Valid")
    p.set(xlim=(100, None), ylim=(None, 3.0))
    return p


def train(model: nn.Module) -> None:
    experiment_name = _gen_experiment_name(model)

    # TODO: remove this for final notebook
    wandb.init("unit-scaling-demo")
    cfg = {
        **_general_config_dict,
        **{
            # "device": "cpu",
            "max_iters": 200,
            "eval_interval": 201,
            "wandb_log": True,
            "wandb_project": "unit-scaling-demo",
            "wandb_run_name": None,
            "experiment_name": experiment_name,
        },
    }

    results = run_training(model, cfg, experiment_name)
    train_df = pd.DataFrame.from_dict(
        {
            "Steps": results["train"]["iters"],
            "Loss": results["train"]["losses"],
        }
    )
    valid_df = pd.DataFrame.from_dict(
        {
            "Steps": results["valid"]["iters"],
            "Loss": results["valid"]["losses"],
        }
    )
    train_df["Train/Valid"] = "Train"
    valid_df["Train/Valid"] = "Valid"
    df = pd.concat([train_df, valid_df])
    df["Format"] = experiment_name
    plot(df)
