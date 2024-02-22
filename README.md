# Utility to check correctness of ZeRO

This script runs training iterations using ZeRO and compares loss values, gradients, and parameters after update with those of PyTorch.

## Usage

Usage: `deepspeed [DEEPSPEED_OPTIONS] compare_loss.py [OPTIONS]`

Options

```
$ python compare_loss.py -h
[2024-02-22 08:38:22,353] [INFO] [real_accelerator.py:191:get_accelerator] Setting ds_accelerator to cuda (auto detect)
usage: compare_loss.py [-h] [--local_rank LOCAL_RANK] [--dtype {torch.bfloat16,torch.float16,torch.float32}]
                       [--zero_stage {0,1,2,3}] [--offload_device {none,cpu,nvme}] [--use_torch_adam] [--rtol RTOL]
                       [--atol ATOL]

DeepSpeed ZeRO unit test.

options:
  -h, --help            show this help message and exit
  --local_rank LOCAL_RANK
                        local rank
  --dtype {torch.bfloat16,torch.float16,torch.float32}
                        data type
  --zero_stage {0,1,2,3}
                        ZeRO stage
  --offload_device {none,cpu,nvme}
                        offload device
  --use_torch_adam      use torch adam optimizer
  --rtol RTOL           relative tolerance
  --atol ATOL           absolute tolerance
```

## Examples

When the number of processes is 2, NCCL produces a deterministic result.
When using `--use_torch_adam`, the script uses the PyTorch Adam optimizer instead of the DeepSpeed optimizer.
This brings a slight difference in the parameter update.
Given `--use_torch_adam` and `--dtype torch.float32`, this script is expected to show that the results from DeepSpeed and PyTorch are exactly the same (no tolerance) for all ZeRO stages.

```
deepspeed --num_gpus=2 compare_loss.py --use_torch_adam --dtype torch.float32 --zero_stage 1
```

When dtype is not `torch.float32`, this compares DeepSpeed's results with results of PyTorch's AMP.
This pass is not fully tested and may not work as expected.
I observed pretty large differences in the results of `torch.float16` and `torch.bfloat16` with `--use_torch_adam`.
You can roughly check the matching by setting `--rtol` and `--atol` to a certain value.

```
deepspeed --num_gpus=2 compare_loss.py --dtype torch.float16 --rtol 0.05 --atol 0.2
```
