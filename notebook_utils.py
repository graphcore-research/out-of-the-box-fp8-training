# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import inspect
import os
import subprocess
import sys
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
    backend_names = [b.__qualname__ for b in getattr(model, "backends", [])]
    print(backend_names)
    if backend_names:
        us = any("unit_scaling" in b for b in backend_names)
        fp8 = any("quantisation" in b for b in backend_names)
        if us and not fp8:
            return f"unit_scaled_gpt"
        if fp8 and not us:
            return f"fp8_gpt"
        if us and fp8:
            return f"unit_scaled_fp8_gpt"
    return "gpt"


data_dir = "nanoGPT/data/shakespeare_char"


def download_train_data():
    cwd = os.getcwd()
    os.chdir(data_dir)
    print(f"Downloading training data to: {data_dir}")
    subprocess.run(["python", "prepare.py"])
    os.chdir(cwd)


def plot(df: pd.DataFrame) -> matplotlib.axes.Axes:
    sns.set_theme()
    p = sns.lineplot(data=df, x="Steps", y="Loss", style="Train/Valid")
    p.set(xlim=(100, None), ylim=(None, 3.0))
    return p


def train(model: nn.Module) -> None:
    experiment_name = _gen_experiment_name(model)

    if not os.path.exists(f"{data_dir}/train.bin"):
        download_train_data()

    # TODO: remove these lines for final notebook (or do we want wandb there?)
    import wandb

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
    wandb.init(project="unit-scaling-demo", config=cfg)

    if experiment_name == "unit_scaled_fp8_gpt":
        cfg.update(
            {
                "learning_rate": 2**-6,
                "min_lr": 2**-6 / 10,
            }
        )

    print(f"Training {experiment_name} ...")
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
    plot(df)
