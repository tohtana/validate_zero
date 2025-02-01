# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import pytest
import os
import torch
import argparse

from transformers import enable_full_determinism

from deepspeed.runtime.zero.offload_config import OffloadDeviceEnum
from deepspeed.accelerator import get_accelerator

from common import bf16_required_version_check, compare_loss


def get_args():
    parser = argparse.ArgumentParser(description='DeepSpeed ZeRO correctness test.')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank')
    parser.add_argument('--dtype', choices=['torch.bfloat16', 'torch.float16', 'torch.float32'], default='torch.float32', help='Data type')
    parser.add_argument('--zero_stage', type=int, choices=[0, 1, 2, 3], default=1, help='ZeRO stage')
    parser.add_argument('--offload_device', choices=['none', 'cpu', 'nvme'], default='none', help='Offload device')
    parser.add_argument('--use_torch_adam', action='store_true', help='Use torch adam optimizer')
    parser.add_argument('--ds_autocast', action='store_true', help='Use autocast with DeepSpeed')
    parser.add_argument('--rtol', type=float, default=0., help='Relative tolerance')
    parser.add_argument('--atol', type=float, default=0., help='Absolute tolerance')
    parser.add_argument('--deterministic', action='store_true', help='Deterministic mode')
    
    return parser.parse_args()


class SimpleModel(torch.nn.Module):

    def __init__(self, hidden_dim, nlayers=1):
        super(SimpleModel, self).__init__()
        self.linears = torch.nn.ModuleList([torch.nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(nlayers)])
        self.loss = torch.nn.MSELoss()

    def forward(self, x, y):
        if len(self.linears) == 1:
            x = self.linears[0](x)
        else:
            for i, l in enumerate(self.linears):
                x = self.linears[i // 2](x) + l(x)
        return self.loss(x, y)
    

def main(args):
    if args.deterministic:
        enable_full_determinism(0)

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

    compare_loss(args, SimpleModel, rtol=args.rtol, atol=args.atol)


if __name__ == '__main__':
    args = get_args()
    main(args)