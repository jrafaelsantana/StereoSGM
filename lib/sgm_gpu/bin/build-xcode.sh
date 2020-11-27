#!/bin/bash

opts=(
    CUDA_TOOLKIT_ROOT_DIR=/Developer/NVIDIA/CUDA-9.0
    CUDA_EXAMPLE_USE_CC_EXT=ON
)

polly.py --toolchain xcode-hid-sections --config Release --fwd ${opts[@]} --verbose ${*}
