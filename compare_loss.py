# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import os
import torch
import argparse

from deepspeed.runtime.zero.offload_config import OffloadDeviceEnum
from deepspeed.accelerator import get_accelerator

from common import bf16_required_version_check, compare_loss


def get_args():
    parser = argparse.ArgumentParser(description='DeepSpeed ZeRO unit test.')
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank')
    parser.add_argument('--dtype', choices=['torch.bfloat16', 'torch.float16', 'torch.float32'], default='torch.float32', help='data type')
    parser.add_argument('--zero_stage', type=int, choices=[0, 1, 2, 3], default=1, help='ZeRO stage')
    parser.add_argument('--offload_device', choices=['none', 'cpu', 'nvme'], default='none', help='offload device')
    parser.add_argument('--use_torch_adam', action='store_true', help='use torch adam optimizer')
    parser.add_argument('--rtol', type=float, default=0., help='relative tolerance')
    parser.add_argument('--atol', type=float, default=0., help='absolute tolerance')
    
    return parser.parse_args()


class SimpleModel(torch.nn.Module):

    def __init__(self, hidden_dim, nlayers=1):
        super(SimpleModel, self).__init__()
        self.linears = torch.nn.ModuleList([torch.nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(nlayers)])
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

    def forward(self, x, y):
        if len(self.linears) == 1:
            x = self.linears[0](x)
        else:
            for i, l in enumerate(self.linears):
                x = self.linears[i // 2](x) + l(x)
        return self.cross_entropy_loss(x, y)
    

def main(args):
    dtype = eval(args.dtype)
    zero_stage = args.zero_stage
    offload_device = eval(f"OffloadDeviceEnum.{args.offload_device}")
    
    get_accelerator().set_device(args.local_rank)

    if dtype == torch.bfloat16 and not bf16_required_version_check():
        pytest.skip(
            " DeepSpeed BFloat16 tests need torch >= 1.10, NCCL >= 2.10.3, CUDA > =11.0 and HW support for BFloat16 to run correctly"
        )

    if offload_device == OffloadDeviceEnum.nvme:
        if zero_stage != 3:
            pytest.skip(f"Nvme offload not supported for zero stage {zero_stage}")

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
    if dtype == torch.float16:
        config_dict["fp16"] = {"enabled": True, "initial_scale_power": 8}
    elif dtype == torch.bfloat16:
        config_dict["bf16"] = {"enabled": True}

    compare_loss(args, SimpleModel, rtol=args.rtol, atol=args.atol)


if __name__ == '__main__':
    args = get_args()
    main(args)