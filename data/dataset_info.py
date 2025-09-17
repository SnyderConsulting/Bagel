# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""Dataset registry for Bagel image editing fine-tuning."""

from .interleave_datasets import UnifiedEditIterableDataset


DATASET_REGISTRY = {
    'unified_edit': UnifiedEditIterableDataset,
}


# Populate this dictionary with your own unified edit datasets. Example entry:
# DATASET_INFO['unified_edit']['my_dataset'] = {
#     "data_dir": "/abs/path/to/parquet_dir",
#     "num_files": 1,
#     "num_total_samples": 52,
#     "parquet_info_path": "/abs/path/to/parquet_info.json",
# }
DATASET_INFO = {
    'unified_edit': {},
}
