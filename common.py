# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import random
import os
import numpy as np
from typing import Callable, Any
from copy import deepcopy

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler

import deepspeed
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.zero import GatheredParameters
from deepspeed.git_version_info import torch_info
from deepspeed.runtime.zero.offload_config import OffloadDeviceEnum



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


def train_amp(baseline_model,
                baseline_optimizer,
                target_engine,
                dtype,
                scaler,
                x, y,
                rtol, atol,
                ds_autocast):
    # Runs the forward pass with autocasting.
    with torch.autocast(device_type="cuda", dtype=dtype):
        baseline_optimizer.zero_grad()
        baseline_loss = baseline_model(x, y)

    scaler.scale(baseline_loss).backward()
    scaler.step(baseline_optimizer)
    scaler.update()

    if not ds_autocast:
        x = x.to(dtype)
        y = y.to(dtype)
    target_loss = target_engine(x, y)

    assert torch.allclose(baseline_loss.float(), target_loss.float(), rtol=rtol, atol=atol)

    target_engine.backward(target_loss)
    target_engine.step()

    with GatheredParameters(target_engine.parameters()):
        for p1, p2 in zip(baseline_model.parameters(), target_engine.parameters()):
            assert torch.allclose(p1.float(), p2.float(), rtol=rtol, atol=atol)


def train_no_amp(baseline_model,
                 baseline_optimizer,
                 target_engine,
                 x, y,
                 rtol, atol):

    baseline_loss = baseline_model(x, y)
    target_loss = target_engine(x, y)

    assert torch.allclose(baseline_loss, target_loss, rtol=rtol, atol=atol)

    with GatheredParameters(target_engine.parameters()):
        for p1, p2 in zip(baseline_model.parameters(), target_engine.parameters()):
            assert torch.allclose(p1, p2, rtol=rtol, atol=atol)

    baseline_loss.backward()
    target_engine.backward(target_loss)

    for p1, p2 in zip(baseline_model.parameters(), target_engine.parameters()):
        g2 = deepspeed.utils.safe_get_full_grad(p2)
        assert torch.allclose(p1, p2, rtol=rtol, atol=atol)
        assert torch.allclose(p1.grad, g2, rtol=rtol, atol=atol)

    baseline_optimizer.step()
    target_engine.step()

    baseline_model.zero_grad()

    with GatheredParameters(target_engine.parameters()):
        for p1, p2 in zip(baseline_model.parameters(), target_engine.parameters()):
            assert torch.allclose(p1, p2, rtol=rtol, atol=atol)


def compare_loss(args, model_cls, rtol=1e-2, atol=1e-2):
    iteration = 5
    hidden_dim = 10

    dtype = eval(args.dtype)
    zero_stage = args.zero_stage
    offload_device = eval(f"OffloadDeviceEnum.{args.offload_device}")
    
    get_accelerator().set_device(args.local_rank)

    if dtype == torch.bfloat16 and not bf16_required_version_check():
        raise ValueError("DeepSpeed BFloat16 tests need torch >= 1.10, NCCL >= 2.10.3, CUDA > =11.0 and HW support for BFloat16 to run correctly")

    if offload_device == OffloadDeviceEnum.nvme:
        if zero_stage != 3:
            raise ValueError(f"Nvme offload not supported for zero stage {zero_stage}")

    config_dict = {
        "train_micro_batch_size_per_gpu": 1,
        "steps_per_print": 1,
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 0.015
            },
        },
        "zero_optimization": {
            "stage": zero_stage,
        },
    }

    if offload_device == OffloadDeviceEnum.cpu:
        config_dict["zero_optimization"]["offload_optimizer"] = {"device": offload_device}
    elif offload_device == OffloadDeviceEnum.nvme:
        tmpdir = os.getcwd()
        config_dict["zero_optimization"]["offload_optimizer"] = {
            "device": offload_device,
            "nvme_path": str(tmpdir)
        }

    if args.ds_autocast:
        config_dict["torch_autocast"] = {"enabled": True, "dtype": str(dtype)}
    else:
        if dtype == torch.float16:
            config_dict["fp16"] = {"enabled": True, "initial_scale_power": 8}
        elif dtype == torch.bfloat16:
            config_dict["bf16"] = {"enabled": True}

    device = torch.device(get_accelerator().current_device_name())
    model = model_cls(hidden_dim)

    deepspeed.init_distributed(dist_backend='nccl')

    i = get_accelerator().current_device()
    lr = config_dict["optimizer"]["params"]["lr"]
    baseline_model = DDP(deepcopy(model).to(device=device, dtype=torch.float32), device_ids=[i], output_device=i)
    baseline_optimizer = torch.optim.AdamW(baseline_model.parameters(), lr=lr, weight_decay=0.0)

    use_amp = dtype != torch.float32
    scaler = GradScaler() if use_amp else None

    stage_3_enabled = config_dict["zero_optimization"]["stage"] == 3
    if stage_3_enabled:
        with deepspeed.zero.Init(config_dict_or_path=config_dict):
            target_model = model_cls(hidden_dim)
        with GatheredParameters(target_model.parameters(), modifier_rank=0):
            for p1, p2 in zip(target_model.parameters(), model.parameters()):
                p1.data.copy_(p2.data)
    else:
        target_model = deepcopy(model)

    if args.use_torch_adam:
        ds_optimizer = torch.optim.Adam(target_model.parameters(), lr=lr)
        del config_dict["optimizer"]
        target_engine, _, _, _ = deepspeed.initialize(config=config_dict,
                                                                     model=target_model,
                                                                     optimizer=ds_optimizer)
    else:
        target_engine, _, _, _ = deepspeed.initialize(config=config_dict,
                                                                     model=target_model,
                                                                     model_parameters=target_model.parameters())

    train_batch_size = config_dict["train_micro_batch_size_per_gpu"]

    xs = [torch.randn(train_batch_size, hidden_dim, device=device, dtype=torch.float32) for _ in range(iteration)]
    ys = [torch.randn_like(x) for x in xs]

    for i, (x, y) in enumerate(zip(xs, ys)):
        if use_amp:
            train_amp(baseline_model, baseline_optimizer, target_engine, dtype, scaler, x, y, rtol, atol, args.ds_autocast)
        else:
            train_no_amp(baseline_model, baseline_optimizer, target_engine, x, y, rtol, atol)
