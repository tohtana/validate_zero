# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import random
import os
import numpy as np
from copy import deepcopy

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.zero import GatheredParameters
from deepspeed.git_version_info import torch_info

from typing import Callable, Any

TIMEOUT = 600


class EnableDeterminism:

    def __init__(self, seed: int):
        local_rank = int(os.getenv("LOCAL_RANK", "0"))

        self.seed = seed + local_rank
        self.saved_random_state = None
        self.saved_np_random_state = None
        self.saved_cuda_launch_blocking = None
        self.saved_cublas_workspace_config = None
        self.saved_deterministic_algorithms = None

    def __enter__(self):
        self.saved_random_state = random.getstate()
        self.saved_np_random_state = np.random.get_state()
        self.saved_acc_rng_state = get_accelerator().get_rng_state()
        self.saved_cuda_launch_blocking = os.environ.get("CUDA_LAUNCH_BLOCKING", "")
        self.saved_cublas_workspace_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG", "")
        self.saved_deterministic_algorithms = torch.are_deterministic_algorithms_enabled()

        random.seed(self.seed)
        np.random.seed(self.seed)
        get_accelerator().manual_seed(self.seed)
        get_accelerator().manual_seed_all(self.seed)

        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        torch.use_deterministic_algorithms(True)

    def __exit__(self, type, value, traceback):
        random.setstate(self.saved_random_state)
        np.random.set_state(self.saved_np_random_state)
        get_accelerator().set_rng_state(self.saved_acc_rng_state)
        os.environ["CUDA_LAUNCH_BLOCKING"] = self.saved_cuda_launch_blocking
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = self.saved_cublas_workspace_config
        torch.use_deterministic_algorithms(self.saved_deterministic_algorithms)


def enable_determinism(seed: int):

    def decorator(func: Callable) -> Callable:

        def wrapper(*args: Any, **kwargs: Any):
            with EnableDeterminism(seed):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def bf16_required_version_check(accelerator_check=True):
    split_version = lambda x: map(int, x.split('.')[:2])
    TORCH_MAJOR, TORCH_MINOR = split_version(torch_info['version'])
    NCCL_MAJOR, NCCL_MINOR = split_version(torch_info['nccl_version'])
    CUDA_MAJOR, CUDA_MINOR = split_version(torch_info['cuda_version'])

    # Sometimes bf16 tests are runnable even if not natively supported by accelerator
    if accelerator_check:
        accelerator_pass = get_accelerator().is_bf16_supported()
    else:
        accelerator_pass = True

    torch_version_available = TORCH_MAJOR > 1 or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10)
    cuda_version_available = CUDA_MAJOR >= 11
    nccl_version_available = NCCL_MAJOR > 2 or (NCCL_MAJOR == 2 and NCCL_MINOR >= 10)
    npu_available = get_accelerator().device_name() == 'npu'

    if torch_version_available and cuda_version_available and nccl_version_available and accelerator_pass:
        return True
    elif npu_available:
        return True
    else:
        return False
    

@enable_determinism(123)
def compare_loss(model_cls, config, dtype, rtol=1e-2, atol=1e-2):
    iteration = 5
    hidden_dim = 10

    device = torch.device(get_accelerator().current_device_name())
    model = model_cls(hidden_dim)

    i = get_accelerator().current_device()
    baseline_model = DDP(deepcopy(model).to(device=device, dtype=torch.float), device_ids=[i], output_device=i)
    baseline_optimizer = torch.optim.Adam(baseline_model.parameters(), lr=config["optimizer"]["params"]["lr"])

    if config["zero_optimization"]["stage"] == 3:
        with deepspeed.zero.Init(config_dict_or_path=config):
            target_model = model_cls(hidden_dim)
        with GatheredParameters(target_model.parameters(), modifier_rank=0):
            for p1, p2 in zip(target_model.parameters(), model.parameters()):
                p1.data.copy_(p2.data)
    else:
        target_model = deepcopy(model)

    target_engine, target_optimizer, _, _ = deepspeed.initialize(config=config,
                                                                 model=target_model,
                                                                 model_parameters=target_model.parameters())

    train_batch_size = config["train_micro_batch_size_per_gpu"]

    xs = [torch.randn(train_batch_size, hidden_dim, device=device, dtype=dtype) for _ in range(iteration)]
    ys = [torch.randn_like(x) for x in xs]

    for x, y in zip(xs, ys):
        baseline_loss = baseline_model(x, y)
        target_loss = target_engine(x, y)

        assert torch.allclose(baseline_loss, target_loss, rtol=rtol, atol=atol)

        baseline_loss.backward()
        target_engine.backward(target_loss)

        baseline_optimizer.step()
        target_optimizer.step()

        with GatheredParameters(target_engine.parameters()):
            for p1, p2 in zip(baseline_model.parameters(), target_engine.parameters()):
                assert torch.allclose(p1.to(dtype), p2, rtol=rtol, atol=atol)

