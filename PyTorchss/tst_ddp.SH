#!/bin/bash
set -ex
torchrun --nproc_per_node=2 --nnodes=1 --master_addr="172.17.0.2" --master_port=50574 tst_ddp.py

