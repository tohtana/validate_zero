#!/bin/bash
set -e

# deepspeed --num_gpus=2 compare_loss.py --use_torch_adam --num_layers 2 --dtype torch.bfloat16 --zero_stage 1 --universal_optimizer
# deepspeed --num_gpus=2 compare_loss.py --use_torch_adam --num_layers 2 --dtype torch.bfloat16 --zero_stage 1 --universal_optimizer --optimizer_dtype torch.float32
# deepspeed --num_gpus=2 compare_loss.py --use_torch_adam --num_layers 2 --dtype torch.bfloat16 --zero_stage 1 --universal_optimizer --optimizer_dtype torch.float32 --grad_accum_dtype torch.float32

deepspeed --num_gpus=2 compare_loss.py --use_torch_adam --num_layers 2 --dtype torch.bfloat16 --zero_stage 2 --universal_optimizer
# deepspeed --num_gpus=2 compare_loss.py --use_torch_adam --num_layers 2 --dtype torch.bfloat16 --zero_stage 2 --universal_optimizer --optimizer_dtype torch.float32
# deepspeed --num_gpus=2 compare_loss.py --use_torch_adam --num_layers 2 --dtype torch.bfloat16 --zero_stage 2 --universal_optimizer --optimizer_dtype torch.float32 --grad_accum_dtype torch.float32