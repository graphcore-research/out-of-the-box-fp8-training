# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import poptorch
import torch
import tqdm
from torch import Tensor, nn
from torch.fx.graph_module import GraphModule

import unit_scaling.transforms.utils

IPU_CONFIG_DEFAULTS = dict(
    compute_batch_size=4,
    replication_factor=4,
)


def prepare_for_ipu(module: nn.Module, example_inputs: List[Tensor]) -> GraphModule:
    """Trace the module on CPU, ready for execution on IPU."""
    graph_module: GraphModule

    def _backend(gm: GraphModule, example_inputs: List[Tensor]) -> GraphModule:
        # The model will be traced on CPU, so we should convert any device="cpu"
        # constant tensors to device="ipu".
        for n in gm.graph.nodes:
            if n.kwargs.get("device") == torch.device("cpu"):
                n.kwargs = {**n.kwargs, "device": torch.device("ipu:0")}
        nonlocal graph_module
        graph_module = gm
        return gm

    module = unit_scaling.transforms.utils.apply_transform(module, _backend)
    # Run a forward pass on CPU to trigger compilation, but use the underlying graph_module
    # on IPU, since TorchDynamo doesn't support IpuTensor
    module(*example_inputs)
    return graph_module


def run_training(
    model: nn.Module, config_dict: Dict[str, Any], experiment_name: str
) -> Dict[str, Dict[str, List[float]]]:
    cfg = Namespace(**{**IPU_CONFIG_DEFAULTS, **config_dict})
    if cfg.batch_size % (cfg.compute_batch_size * cfg.replication_factor) != 0:
        raise ValueError(
            f"Batch size {cfg.batch_size} not divisible by"
            " compute_batch_size * replication_factor"
            f" = {cfg.compute_batch_size} * {cfg.replication_factor}"
        )

    data_dir = Path("nanoGPT/data", cfg.dataset)
    data = {
        split: torch.frombuffer(
            (data_dir / f"{split}.bin").read_bytes(), dtype=torch.int16
        )
        for split in ["train", "val"]
    }

    def get_batch(split: str) -> Tuple[Tensor, Tensor]:
        idx = torch.randint(len(data[split]) - cfg.block_size, (cfg.batch_size,))
        tokens = torch.stack([data[split][i : i + cfg.block_size] for i in idx]).to(
            torch.long
        )
        return tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()

    def lr_schedule_fn(step: int) -> float:
        if step < cfg.warmup_iters:
            return step / cfg.warmup_iters
        min_ratio = cfg.min_lr / cfg.learning_rate
        progress = (step - cfg.warmup_iters) / (cfg.max_iters - cfg.warmup_iters)
        return min_ratio + (1 - min_ratio) * (0.5 + 0.5 * np.cos(np.pi * progress))

    model = prepare_for_ipu(
        model, [t[: cfg.compute_batch_size] for t in get_batch("val")]
    )
    opt = poptorch.optim.Adam(
        model.parameters(), lr=cfg.learning_rate, betas=(0.9, 0.99)
    )
    lr_schedule = torch.optim.lr_scheduler.LambdaLR(opt, lr_schedule_fn)
    options = poptorch.Options()
    options.replicationFactor(cfg.replication_factor)
    options.outputMode(poptorch.OutputMode.All)
    training_options, inference_options = options.clone(), options.clone()
    iterations = cfg.batch_size // (cfg.compute_batch_size * options.replication_factor)
    training_options.Training.gradientAccumulation(iterations)
    inference_options.deviceIterations(iterations)
    trainer = poptorch.trainingModel(model, options=training_options, optimizer=opt)
    evaluator = poptorch.inferenceModel(model, options=inference_options)

    results = {
        "train": {"iters": [], "losses": []},
        "valid": {"iters": [], "losses": []},
    }
    iter_num = 0

    def step() -> None:
        nonlocal iter_num
        if iter_num % cfg.eval_interval == 0 and cfg.eval_iters:
            if iter_num:
                trainer.detachFromDevice()
            losses = [evaluator(*get_batch("val"))[1] for _ in range(cfg.eval_iters)]
            results["valid"]["losses"].append(float(torch.mean(torch.stack(losses))))
            results["valid"]["iters"].append(iter_num)
            evaluator.detachFromDevice()
        loss = float(torch.mean(trainer(*get_batch("train"))[1]))
        results["train"]["losses"].append(loss)
        results["train"]["iters"].append(iter_num)
        lr_schedule.step()
        trainer.setOptimizer(opt)
        iter_num += 1

    try:
        step()  # trigger compilation before starting tqdm
        for _ in tqdm.tqdm(range(1, cfg.max_iters), initial=1, total=cfg.max_iters):
            step()
        return results
    finally:
        trainer.destroy()
