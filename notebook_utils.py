# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import inspect
import os
import subprocess
from typing import Any, Dict

import matplotlib.axes
import pandas as pd
import seaborn as sns
import wandb
from torch import nn

from nanoGPT.config import train_shakespeare_char
from nanoGPT.model import GPTConfig

try:
    from train_ipu import run_training

except ImportError:  # not on IPU...
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


def train(model: nn.Module, **config_overrides: Any) -> pd.DataFrame:
    experiment_name = _gen_experiment_name(model) + config_overrides.pop(
        "experiment_name_suffix", ""
    )

    if not os.path.exists(f"{data_dir}/train.bin"):
        download_train_data()

    cfg = _general_config_dict.copy()
    cfg.update(
        eval_interval=1000,
        wandb_log=True,
        wandb_project="unit-scaling-demo",
        experiment_name=experiment_name,
    )
    # if experiment_name == "unit_scaled_fp8_gpt":
    #     cfg.update(learning_rate=2**-6, min_lr=2**-6 / 10)
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
    train_df["Train/Valid"] = "Train"
    valid_df["Train/Valid"] = "Valid"
    df = pd.concat([train_df, valid_df])
    plot(df)
    return df


# TODO: improve
import pickle

from torch import LongTensor, Tensor

import example_text


def example_seqs(batch_size):
    text = example_text.hansard_lords_1964
    seq_len = len(text) // batch_size
    return [text[i * seq_len : (i + 1) * seq_len] for i in range(batch_size)]


def create_model_inputs(
    tokenize_fn,
    example_seqs,
    max_seq_len: int,
):
    # Although we can't crop the example sequences exactly based on `max_seq_len`
    # (we don't know how many tokens will be produced), we assume that the tokenizer
    # will produce > (len(pre_tok_seq) / 10) tokens and crop accordingly.
    # his just avoids the tokenizer having to do too much unnecessary work
    pre_tok_seqs = [pre_tok_seq[: 10 * max_seq_len] for pre_tok_seq in example_seqs]
    # try:
    #     seqs = tokenize_fn(pre_tok_seqs)
    # except TypeError:
    seqs = [tokenize_fn(s) for s in pre_tok_seqs]

    seq_len = min(max_seq_len, max(len(s) for s in seqs)) + 1

    # The tokenize function may produce a ready tensor. However in most cases it gives a
    # list. This means we have to turn this into a tensor manually
    if not isinstance(seqs, Tensor):
        truncated_seqs = []
        for s in seqs:
            new_s = s[:seq_len]
            if len(new_s) < seq_len:  # handles padding (unlikely wth example txt)
                new_s += tokenize_fn(" ")[0] * (seq_len - len(new_s))
            truncated_seqs.append(new_s)
        seqs = LongTensor(truncated_seqs)
    input_idxs = seqs[:, : seq_len - 1].clone()
    labels = seqs[:, 1:seq_len].clone()
    return input_idxs, labels


def demo_data(batch_size=64):
    with open("nanoGPT/data/shakespeare_char/meta.pkl", "rb") as f:
        meta = pickle.load(f)
        stoi = meta["stoi"]

    def encode(s):
        return [stoi.get(c, stoi[" "]) for c in s]

    seqs = example_seqs(batch_size)
    inputs, labels = create_model_inputs(
        encode,
        seqs,
        max_seq_len=_general_config_dict["block_size"],
    )
    return inputs, labels
