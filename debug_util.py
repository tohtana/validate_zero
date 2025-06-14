# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import os
import torch
import logging


def setup_compilation_debugging():
    """Setup enhanced debugging for torch.compile recompilations"""
    # CRITICAL: Set this BEFORE torch._logging.set_logs
    os.environ["TORCH_LOGS"] = "+recompiles,+graph_breaks,+guards,+dynamo,+inductor,+distributed"
    os.environ["TORCH_LOGS"] = "+recompiles"
    os.environ["TORCH_COMPILE_DEBUG"] = "1" 
    os.environ["TORCHDYNAMO_VERBOSE"] = "1"
    os.environ["TORCHINDUCTOR_VERBOSE"] = "1"
    
    # Enable comprehensive logging with VERBOSE levels
    torch._logging.set_logs(
        recompiles=logging.DEBUG,
        graph_breaks=logging.DEBUG, 
        guards=logging.DEBUG,
        dynamo=logging.DEBUG,
        inductor=logging.DEBUG,
        distributed=logging.DEBUG,
        output_code=True,
        schedule=True,
        trace_source=True,
        trace_call=True
    )
    
    # Setup Python loggers for torch internals
    for logger_name in [
        'torch._dynamo',
        'torch._dynamo.guards', 
        'torch._dynamo.convert_frame',
        'torch._dynamo.eval_frame',
        'torch._inductor'
    ]:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('[%(name)s] %(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        if not logger.handlers:
            logger.addHandler(handler)
    
    print("🔧 Enhanced torch.compile debugging enabled")
    print("Environment variables set:")
    for key in ["TORCH_LOGS", "TORCH_COMPILE_DEBUG", "TORCHDYNAMO_VERBOSE", "TORCHINDUCTOR_VERBOSE"]:
        print(f"  {key}={os.environ.get(key, 'not set')}")


def debug_guard_failures():
    """Setup additional guard failure debugging"""
    import torch._dynamo
    import torch._dynamo.config
    
    # Enable maximum verbosity for guard failures - only valid config options
    try:
        torch._dynamo.config.verbose = True
        print("✓ Set verbose = True")
    except Exception as e:
        print(f"⚠ Could not set verbose: {e}")
    
    print("✓ Guard failure debugging configured")
