#!/bin/bash
# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
# Script to be sourced on launch of the Gradient Notebook

DETECTED_NUMBER_OF_IPUS=$(python .gradient/available_ipus.py)
if [[ "$1" == "test" ]]; then
    IPU_ARG="${DETECTED_NUMBER_OF_IPUS}"
else
    IPU_ARG=${1:-"${DETECTED_NUMBER_OF_IPUS}"}
fi

export NUM_AVAILABLE_IPU=${IPU_ARG}
export GRAPHCORE_POD_TYPE="pod${IPU_ARG}"

export POPLAR_EXECUTABLE_CACHE_DIR="/tmp/exe_cache"
export DATASET_DIR="/tmp/dataset_cache"
export CHECKPOINT_DIR="/tmp/checkpoints"

# mounted public dataset directory (path in the container)
# in the Paperspace environment this would be ="/datasets"
export PUBLIC_DATASET_DIR="/datasets"

export POPTORCH_CACHE_DIR="${POPLAR_EXECUTABLE_CACHE_DIR}"
export POPTORCH_LOG_LEVEL=ERR
export RDMAV_FORK_SAFE=1

export PIP_DISABLE_PIP_VERSION_CHECK=1 CACHE_DIR=/tmp
if [ -z "${SETUP_NO_JUPYTER}" ]; then
    jupyter lab --allow-root --ip=0.0.0.0 --no-browser --ServerApp.trust_xheaders=True \
                --ServerApp.disable_check_xsrf=False --ServerApp.allow_remote_access=True \
                --ServerApp.allow_origin='*' --ServerApp.allow_credentials=True
fi
