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


def enable_full_determinism(seed: int):
    """Enable full determinism for reproducible results"""
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    final_seed = seed + local_rank
    
    print(f"Setting random seed to {final_seed} for local rank {local_rank}")

    # Set all random seeds
    random.seed(final_seed)
    np.random.seed(final_seed)
    torch.manual_seed(final_seed)
    get_accelerator().manual_seed(final_seed)
    get_accelerator().manual_seed_all(final_seed)

    # Set CUDA/PyTorch determinism settings
    os.environ["PYTHONHASHSEED"] = "0"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('high')

    # Enable CUDNN deterministic mode
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Handle torch._inductor if available (for newer PyTorch versions)
    try:
        torch._inductor.config.fallback_random = True
        torch._inductor.config.max_autotune = False
    except AttributeError:
        # Older PyTorch versions may not have these settings
        pass


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
                x_batch, y_batch,
                gradient_accumulation_steps,
                rtol, atol):
    # Runs the forward pass with autocasting and gradient accumulation.
    baseline_loss_total = 0.0
    target_loss_total = 0.0
    
    baseline_optimizer.zero_grad()
    
    for step in range(gradient_accumulation_steps):
        x = x_batch[step]
        y = y_batch[step]
        
        # Disable gradient synchronization for all but the last micro-batch
        is_last_step = (step == gradient_accumulation_steps - 1)
        
        if is_last_step:
            # Normal forward/backward with gradient synchronization
            with torch.autocast(device_type="cuda", dtype=dtype):
                baseline_loss_raw = baseline_model(x, y)
                baseline_loss_scaled = baseline_loss_raw / gradient_accumulation_steps
                baseline_loss_total += baseline_loss_raw.item()  # Accumulate raw loss for comparison
                scaler.scale(baseline_loss_scaled).backward()  # Backward on scaled loss for correct gradients
        else:
            # Forward/backward without gradient synchronization
            with baseline_model.no_sync():
                with torch.autocast(device_type="cuda", dtype=dtype):
                    baseline_loss_raw = baseline_model(x, y)
                    baseline_loss_scaled = baseline_loss_raw / gradient_accumulation_steps
                    baseline_loss_total += baseline_loss_raw.item()  # Accumulate raw loss for comparison
                    scaler.scale(baseline_loss_scaled).backward()  # Backward on scaled loss for correct gradients
        
        target_loss = target_engine(x.to(dtype), y.to(dtype))
        target_loss_total += target_loss.item()
        target_engine.backward(target_loss)
    
    scaler.step(baseline_optimizer)
    scaler.update()
    target_engine.step()

    # Compare accumulated losses
    # Both baseline_loss_total and target_loss_total now contain raw (unscaled) losses
    assert torch.allclose(torch.tensor(baseline_loss_total).half(), torch.tensor(target_loss_total), rtol=rtol, atol=atol)

    with GatheredParameters(target_engine.parameters()):
        for i, (p1, p2) in enumerate(zip(baseline_model.parameters(), target_engine.parameters())):
            p1_half = p1.half()
            if not torch.allclose(p1_half, p2, rtol=rtol, atol=atol):
                max_diff = torch.max(torch.abs(p1_half - p2)).item()
                mean_diff = torch.mean(torch.abs(p1_half - p2)).item()
                print(f"Parameter {i} mismatch (AMP):")
                print(f"  Max absolute difference: {max_diff}")
                print(f"  Mean absolute difference: {mean_diff}")
                print(f"  Tolerance settings: rtol={rtol}, atol={atol}")
                print(f"  Parameter shapes: baseline={p1_half.shape}, target={p2.shape}")
                print(f"  Baseline param stats: min={p1_half.min().item():.6f}, max={p1_half.max().item():.6f}, mean={p1_half.mean().item():.6f}")
                print(f"  Target param stats: min={p2.min().item():.6f}, max={p2.max().item():.6f}, mean={p2.mean().item():.6f}")
            assert torch.allclose(p1_half, p2, rtol=rtol, atol=atol), f"Parameter {i} comparison failed (AMP)"


def train_no_amp(baseline_model,
                 baseline_optimizer,
                 target_engine,
                 x_batch, y_batch,
                 gradient_accumulation_steps,
                 rtol, atol):

    baseline_loss_total = 0.0
    target_loss_total = 0.0
    
    baseline_optimizer.zero_grad()
    
    for step in range(gradient_accumulation_steps):
        x = x_batch[step]
        y = y_batch[step]
        
        # Disable gradient synchronization for all but the last micro-batch
        is_last_step = (step == gradient_accumulation_steps - 1)
        
        if is_last_step:
            # Normal forward/backward with gradient synchronization
            baseline_loss_raw = baseline_model(x, y)
            baseline_loss_scaled = baseline_loss_raw / gradient_accumulation_steps
            baseline_loss_total += baseline_loss_raw.item()  # Accumulate raw loss for comparison
            baseline_loss_scaled.backward()  # Backward on scaled loss for correct gradients
        else:
            # Forward/backward without gradient synchronization
            with baseline_model.no_sync():
                baseline_loss_raw = baseline_model(x, y)
                baseline_loss_scaled = baseline_loss_raw / gradient_accumulation_steps
                baseline_loss_total += baseline_loss_raw.item()  # Accumulate raw loss for comparison
                baseline_loss_scaled.backward()  # Backward on scaled loss for correct gradients
        
        target_loss = target_engine(x, y)
        target_loss_total += target_loss.item()
        # DeepSpeed handles the scaling internally, so we don't need to scale the loss here
        target_engine.backward(target_loss)
        # The gradient accumulation boundary is also handled by DeepSpeed
        target_engine.step()

    baseline_optimizer.step()

    # Compare accumulated losses
    # Both baseline_loss_total and target_loss_total now contain raw (unscaled) losses
    print(f"Baseline loss: {baseline_loss_total}, Target loss: {target_loss_total} atol={atol}, rtol={rtol}")
    assert torch.allclose(torch.tensor(baseline_loss_total), torch.tensor(target_loss_total), rtol=rtol, atol=atol)

    with GatheredParameters(target_engine.parameters()):
        for i, (p1, p2) in enumerate(zip(baseline_model.parameters(), target_engine.parameters())):
            if not torch.allclose(p1, p2, rtol=rtol, atol=atol):
                max_diff = torch.max(torch.abs(p1 - p2)).item()
                mean_diff = torch.mean(torch.abs(p1 - p2)).item()
                print(f"Parameter {i} mismatch:")
                print(f" p1 {p1} p2 {p2}")
                print(f"  Max absolute difference: {max_diff}")
                print(f"  Mean absolute difference: {mean_diff}")
                print(f"  Tolerance settings: rtol={rtol}, atol={atol}")
                print(f"  Parameter shapes: baseline={p1.shape}, target={p2.shape}")
                print(f"  Baseline param stats: min={p1.min().item():.6f}, max={p1.max().item():.6f}, mean={p1.mean().item():.6f}")
                print(f"  Target param stats: min={p2.min().item():.6f}, max={p2.max().item():.6f}, mean={p2.mean().item():.6f}")
            assert torch.allclose(p1, p2, rtol=rtol, atol=atol), f"Parameter {i} comparison failed"


def compare_loss(args, model_cls, rtol=1e-2, atol=1e-2):
    iteration = args.iterations
    hidden_dim = 4
    gradient_accumulation_steps = args.gradient_accumulation_steps

    dtype = eval(args.dtype)
    zero_stage = args.zero_stage
    offload_device = eval(f"OffloadDeviceEnum.{args.offload_device}")
    
    get_accelerator().set_device(args.local_rank)

    if dtype == torch.bfloat16 and not bf16_required_version_check():
        raise ValueError("DeepSpeed BFloat16 tests need torch >= 1.10, NCCL >= 2.10.3, CUDA > =11.0 and HW support for BFloat16 to run correctly")

    if offload_device == OffloadDeviceEnum.nvme:
        if zero_stage != 3:
            raise ValueError(f"Nvme offload not supported for zero stage {zero_stage}")

    # Enable full determinism
    enable_full_determinism(123)

    # Initialize distributed BEFORE setting deterministic seeds
    deepspeed.init_distributed(dist_backend='nccl')
    
    print(f"Running correctness test with gradient_accumulation_steps={gradient_accumulation_steps}")
    print(f"Running {iteration} iterations with {gradient_accumulation_steps * iteration} total micro steps")
    
    # Now apply deterministic settings
    config_dict = {
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": gradient_accumulation_steps,
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
        "compile": {
            "deepcompile": args.deepcompile
        }
    }

    if offload_device == OffloadDeviceEnum.cpu:
        config_dict["zero_optimization"]["offload_optimizer"] = {"device": offload_device}
    elif offload_device == OffloadDeviceEnum.nvme:
        tmpdir = os.getcwd()
        config_dict["zero_optimization"]["offload_optimizer"] = {
            "device": offload_device,
            "nvme_path": str(tmpdir)
        }

    if dtype == torch.float16:
        config_dict["fp16"] = {"enabled": True, "initial_scale_power": 8}
    elif dtype == torch.bfloat16:
        config_dict["bf16"] = {"enabled": True}

    device = torch.device(get_accelerator().current_device_name())
    model = model_cls(hidden_dim)
    
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

    # Only compile models if requested via command line
    if args.compile:
        target_engine.compile()
        if not args.deepcompile:
            # We don't compile the baseline model if deepcompile is requested as DeepCompile breaks Dynamo's state
            baseline_model.compile()

    train_batch_size = config_dict["train_micro_batch_size_per_gpu"]

    # Generate data for each iteration and gradient accumulation step
    xs = []
    ys = []
    for i in range(iteration):
        x_batch = [torch.randn(train_batch_size, hidden_dim, device=device, dtype=torch.float32) 
                   for _ in range(gradient_accumulation_steps)]
        y_batch = [torch.randn_like(x) for x in x_batch]
        xs.append(x_batch)
        ys.append(y_batch)

    for i, (x_batch, y_batch) in enumerate(zip(xs, ys)):
        if use_amp:
            train_amp(baseline_model, baseline_optimizer, target_engine, dtype, scaler, 
                     x_batch, y_batch, gradient_accumulation_steps, rtol, atol)
        else:
            train_no_amp(baseline_model, baseline_optimizer, target_engine, 
                        x_batch, y_batch, gradient_accumulation_steps, rtol, atol)
