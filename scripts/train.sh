# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

# replace the variables with your own
torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node=1 \
  --master_addr=127.0.0.1 \
  --master_port=1234 \
  train/pretrain_unified_navit.py \
  --dataset_config_file ./data/configs/example.yaml \
  --layer_module Qwen2MoTDecoderLayer \
  --vae_path flux/vae/ae.safetensors \
  --vit_path hf/siglip-so400m-14-980-flash-attn2-navit \
  --llm_path hf/Qwen2.5-0.5B-Instruct \
  --use_flex True \
  --resume_from models/BAGEL-7B-MoT \
  --results_dir results \
  --checkpoint_dir results/checkpoints \
  --max_latent_size 64  \
  --num_workers 1 # use small num_workers since the num_used_data (10) are not enough to split
