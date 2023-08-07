# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import json
import os
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import poptorch
import torch
import torch.nn.functional as F
import tqdm
import wandb
from torch import Tensor, nn
from torch.fx.graph_module import GraphModule

import unit_scaling.functional as U
import unit_scaling.transforms.utils

IPU_CONFIG_DEFAULTS = dict(
    compute_batch_size=4,
    replication_factor=4,
    profile=False,
)


def gradient_accumulation_loss(
    loss: Tensor, batch_size: int, compute_batch_size: int
) -> Tensor:
    loss = loss * compute_batch_size / batch_size
    return poptorch.identity_loss(loss, reduction="none")


def unit_scaled_gradient_accumulation_loss(
    loss: Tensor, batch_size: int, compute_batch_size: int
) -> Tensor:
    loss = U.scale_fwd(loss, compute_batch_size / batch_size)
    return poptorch.identity_loss(loss, reduction="none")


def prepare_for_ipu(
    module: nn.Module, example_inputs: List[Tensor], batch_size: int
) -> GraphModule:
    """Trace the module on CPU, ready for execution on IPU."""
    graph_module: GraphModule

    def _backend(gm: GraphModule, example_inputs: List[Tensor]) -> GraphModule:
        nonlocal graph_module
        graph_module = gm

        # Gradient accumulation sums loss gradients over multiple compute batches.
        # Here we apply a scaling factor to each compute batch loss to get the sum()
        # to equal the global batch loss, behaving the same as plain training
        # (without gradient accumulation).
        loss_replacement_map = {
            F.cross_entropy: gradient_accumulation_loss,
            U.cross_entropy: unit_scaled_gradient_accumulation_loss,
        }
        loss = None
        for n in graph_module.graph.nodes:
            if n.target in loss_replacement_map:
                with graph_module.graph.inserting_after(n):
                    loss = graph_module.graph.call_function(
                        loss_replacement_map[n.target],
                        args=(n,),
                        kwargs=dict(
                            batch_size=batch_size,
                            compute_batch_size=example_inputs[0].shape[0],
                        ),
                    )
            if n.target == "output":
                assert loss, "expected to find a 'cross_entropy' node before 'output'"
                n.replace_input_with(n.args[0][1], loss)

        graph_module.graph.lint()
        graph_module.recompile()
        return graph_module

    module = unit_scaling.transforms.utils.apply_transform(module, _backend)
    # Run a forward pass on CPU to trigger compilation, but use the underlying graph_module
    # on IPU, since TorchDynamo doesn't support IpuTensor
    module(*example_inputs)

    # The model was traced on CPU, so we should convert any device="cpu" constant tensors
    # to device="ipu".
    for n in graph_module.graph.nodes:
        if n.kwargs.get("device") == torch.device("cpu"):
            n.kwargs = {**n.kwargs, "device": torch.device("ipu:0")}
    graph_module.recompile()

    return graph_module


def run_training(
    model: nn.Module, config_dict: Dict[str, Any]
) -> Dict[str, Dict[str, List[float]]]:
    cfg = Namespace(
        **{**IPU_CONFIG_DEFAULTS, **config_dict}, model=model.config.__dict__
    )
    if cfg.wandb_log:
        wandb.init(project=cfg.wandb_project, config=cfg.__dict__, reinit=True)

    if cfg.profile:
        profile = Path("profiles/latest")
        profile.mkdir(exist_ok=True, parents=True)
        os.environ["POPLAR_ENGINE_OPTIONS"] = json.dumps(
            {
                "autoReport.all": True,
                "autoReport.outputArchive": False,
                "autoReport.directory": str(profile),
            }
        )
        (profile / "app.json").write_text(json.dumps(cfg.__dict__))

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
        idx = torch.randint(len(data[split]) - cfg.block_size - 1, (cfg.batch_size,))
        tokens = torch.stack([data[split][i : i + cfg.block_size + 1] for i in idx]).to(
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
        model,
        [t[: cfg.compute_batch_size] for t in get_batch("val")],
        cfg.batch_size,
    )
    options = poptorch.Options()
    options.replicationFactor(cfg.replication_factor)
    options.outputMode(poptorch.OutputMode.All)
    training_options, inference_options = options.clone(), options.clone()
    iterations = cfg.batch_size // (cfg.compute_batch_size * options.replication_factor)
    training_options.Training.gradientAccumulation(iterations)
    inference_options.deviceIterations(iterations)
    opt = poptorch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        betas=(0.9, cfg.beta2),
        # loss_scaling=cfg.compute_batch_size / cfg.batch_size,  # match CPU/GPU behaviour
    )
    lr_schedule = torch.optim.lr_scheduler.LambdaLR(opt, lr_schedule_fn)
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
            val_loss = float(torch.sum(torch.stack(losses)))
            results["valid"]["losses"].append(val_loss)
            results["valid"]["iters"].append(iter_num)
            if cfg.wandb_log:
                wandb.log(dict(val_loss=val_loss, step=iter_num), step=iter_num)
            evaluator.detachFromDevice()
        if iter_num < cfg.max_iters:
            loss = float(torch.sum(trainer(*get_batch("train"))[1]))
            results["train"]["losses"].append(loss)
            results["train"]["iters"].append(iter_num)
            if cfg.wandb_log:
                wandb.log(dict(loss=loss, step=iter_num), step=iter_num)
            lr_schedule.step()
            trainer.setOptimizer(opt)
            iter_num += 1

    try:
        step()  # trigger compilation before starting tqdm
        # +1 iteration for final validation only
        for _ in tqdm.tqdm(list(range(1, cfg.max_iters + 1))):
            step()
        if cfg.wandb_log:
            wandb.finish()
        return results
    except Exception as e:
        if cfg.wandb_log:
            wandb.run.summary["error"] = str(e)
            wandb.finish(1)
        raise
    finally:
        trainer.destroy()
