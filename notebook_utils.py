# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import inspect
import logging
import os
import pickle
import subprocess
from typing import Any, Dict

import matplotlib.axes
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from torch import cuda, nn

from nanoGPT.config import train_shakespeare_char
from nanoGPT.model import GPTConfig

try:
    from train_ipu import run_training

    device = "ipu"
except ImportError:  # not on IPU...
    from train import run_training

    device = "cuda" if cuda.is_available() else "cpu"


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
for key, value in _general_config_dict.items():
    setattr(config, key, value)


def _gen_experiment_name(model: nn.Module) -> str:
    backend_names = [b.__qualname__ for b in getattr(model, "backends", [])]
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
    print(f"Downloading training data/tokenizer to: {data_dir}")
    subprocess.run(["python", "prepare.py"])
    os.chdir(cwd)


class NanoGPTTokenizer:
    def __init__(self):
        meta_file = "nanoGPT/data/shakespeare_char/meta.pkl"
        if not os.path.exists(meta_file):
            download_train_data()

        with open(meta_file, "rb") as f:
            meta = pickle.load(f)
            stoi = meta["stoi"]
            self.encode_fn = lambda s: [stoi.get(c, stoi[" "]) for c in s]

    @property
    def pad_token(self):
        return self.encode_fn(" ")[0]

    def __call__(self, seqs, max_length, *args, **kwargs):
        batch = []
        for s in seqs:
            new_s = self.encode_fn(s)[:max_length]
            if len(new_s) < max_length:
                new_s += self.pad_token * (max_length - len(new_s))
            batch.append(new_s)
        batch = torch.tensor(batch)
        return {
            "input_ids": batch,
            "attention_mask": torch.ones_like(batch),  # nanoGPT ignores this anyway
        }


def plot(df: pd.DataFrame, name: str) -> matplotlib.axes.Axes:
    sns.set_theme()
    ax = sns.lineplot(
        data=df,
        x="Steps",
        y="Loss",
        style="Train/Valid",
        label=name,
        solid_joinstyle="miter",
        solid_capstyle="butt",
        linewidth=1.5,
    )
    ax.set(xlim=(0, None), ylim=(0.6, 3.0))

    # remove duplicate legend entries
    entries = {}
    for h, l in zip(*ax.get_legend_handles_labels()):
        if l not in entries:
            entries[l] = h
    entries[""] = plt.Line2D([0], [0], marker="none", linestyle="none", color="none")
    entries["validation"] = entries.pop("validation")  # move to bottom
    entries["training"] = entries.pop("training")  # move to bottom
    ax.legend(
        entries.values(), entries.keys(), bbox_to_anchor=(1.05, 1), loc="upper left"
    )
    return ax


def train(model: nn.Module, **config_overrides: Any) -> pd.DataFrame:
    if device == "cpu":
        logging.warning(
            "CPU does not have sufficient FLOP/s for tractable training. "
            "Please try again using an IPU or GPU."
        )

    experiment_name = _gen_experiment_name(model) + config_overrides.pop(
        "experiment_name_suffix", ""
    )

    if not os.path.exists(f"{data_dir}/train.bin"):
        download_train_data()

    cfg = _general_config_dict.copy()
    cfg.update(
        device=device,
        experiment_name=experiment_name,
    )
    if "unit_scaled" in experiment_name:
        cfg.update(learning_rate=0.02, min_lr=0.002)
    cfg.update(config_overrides)

    print(f"Training {experiment_name} ...")
    results = run_training(model, cfg)
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
    train_df["Train/Valid"] = "training"
    valid_df["Train/Valid"] = "validation"
    df = pd.concat([train_df, valid_df])
    df["Model"] = experiment_name
    plot(df, experiment_name)
