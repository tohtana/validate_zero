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
                       [--atol ATOL] [--compile] [--deepcompile] [--verbose_logging]

DeepSpeed ZeRO correctness test.

options:
  -h, --help            show this help message and exit
  --local_rank LOCAL_RANK
                        Local rank
  --dtype {torch.bfloat16,torch.float16,torch.float32}
                        Data type
  --zero_stage {0,1,2,3}
                        ZeRO stage
  --offload_device {none,cpu,nvme}
                        Offload device
  --use_torch_adam      Use torch adam optimizer
  --rtol RTOL           Relative tolerance
  --atol ATOL           Absolute tolerance
  --compile             Enable torch.compile() on the model
  --deepcompile         Enable deepcompile optimization
  --verbose_logging     Enable verbose debugging for recompilations
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

**Note**: For debugging torch.compile recompilation issues, see the [Debugging Options](#debugging-options) section below.

## Compilation Options

The script now supports PyTorch compilation features:

### Using torch.compile()

Enable standard PyTorch compilation for both baseline and target models:

```
deepspeed --num_gpus=2 compare_loss.py --compile --use_torch_adam --dtype torch.float32 --zero_stage 1
```

### Using DeepSpeed DeepCompile

Enable DeepSpeed's DeepCompile optimization, which provides compiler-level optimizations for distributed training:

```
deepspeed --num_gpus=2 compare_loss.py --compile --use_torch_adam --deepcompile --zero_stage 3 --dtype torch.bfloat16
```

## Debugging Options

### Verbose Logging for torch.compile

Use the `--verbose_logging` flag to enable detailed debugging information when using `torch.compile()`. This is particularly useful for analyzing compiler's behavior and guard failures:

```
deepspeed --num_gpus=1 compare_loss.py --compile --verbose_logging --zero_stage 1
```

This option enables:
- Comprehensive logging of compiler events
- Guard failure analysis and debugging
- Environment variable setup for maximum torch.compile debugging
- Detailed output of compilation internals

**Note**: Verbose logging produces significant output and should be used primarily for debugging purposes.
