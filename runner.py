from unittest.mock import patch

import numpy as np
import wandb
import yaml
from unit_scaling.functional import residual_add, residual_split
from unit_scaling.transforms import unit_scale

from nanoGPT.model import GPT
from notebook_utils import config, train

# with open("./sweep_config.yaml") as file:
#     cfg = yaml.load(file, Loader=yaml.FullLoader)

# wandb.init(project="unit-scaling-demo", config=cfg)

# config.bias = False
model = unit_scale(GPT(config))
train(
    model,
    eval_iters=0,
    experiment_name_suffix="",
    seed=np.random.randint(2**31),
)
