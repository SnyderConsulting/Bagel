# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import logging
import math
import random

import torch


logger = logging.getLogger(__name__)


class DistributedIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset_name, local_rank=0, world_size=1, num_workers=8):
        self.dataset_name = dataset_name
        self.local_rank = local_rank
        self.world_size = world_size
        self.num_workers = num_workers
        self.rng = random.Random()
        self.data_paths = None

    def get_data_paths(self, *args, **kwargs):
        raise NotImplementedError

    def set_epoch(self, seed=42):
        if self.data_paths is None:
            return

        if isinstance(self.data_paths[0], tuple):
            data_paths = sorted(self.data_paths, key=lambda x: (x[0], x[1]))
        elif isinstance(self.data_paths[0], str):
            data_paths = sorted(self.data_paths)
        else:
            raise ValueError(f"Unknown data_paths type: {type(self.data_paths[0])}")

        self.rng.seed(seed)
        self.rng.shuffle(data_paths)

        total_files = len(data_paths)
        if total_files == 0:
            self.num_files_per_rank = 0
            self.data_paths_per_rank = []
            return

        if self.world_size <= 0:
            raise ValueError("world_size must be a positive integer")

        if total_files < self.world_size:
            logger.warning(
                "Dataset %s has %d shards but world_size=%d; reusing shards to cover all ranks.",
                self.dataset_name,
                total_files,
                self.world_size,
            )
            repeat_factor = math.ceil(self.world_size / total_files)
            data_paths = data_paths * repeat_factor
            self.rng.shuffle(data_paths)

        per_rank_splits = self._balanced_split(data_paths, self.world_size, ensure_non_empty=True)
        self.data_paths_per_rank = per_rank_splits[self.local_rank]
        self.num_files_per_rank = len(self.data_paths_per_rank)

    def get_data_paths_per_worker(self):
        if self.data_paths is None:
            return None

        info = torch.utils.data.get_worker_info()
        if info is None:
            # Single worker: Use all files assigned to the rank
            return self.data_paths_per_rank, 0

        worker_id = info.id
        if info.num_workers <= 0:
            raise ValueError("num_workers must be a positive integer")

        per_worker_splits = self._balanced_split(
            self.data_paths_per_rank,
            info.num_workers,
            ensure_non_empty=bool(self.data_paths_per_rank),
        )
        data_paths_per_worker = per_worker_splits[worker_id]

        return data_paths_per_worker[::-1], worker_id

    def __iter__(self):
        raise NotImplementedError

    def _balanced_split(self, items, num_splits, ensure_non_empty=False):
        if num_splits <= 0:
            raise ValueError("num_splits must be a positive integer")

        items = list(items)
        if not items:
            return [[] for _ in range(num_splits)]

        if ensure_non_empty and len(items) < num_splits:
            repeat_factor = math.ceil(num_splits / len(items))
            items = items * repeat_factor

        base, remainder = divmod(len(items), num_splits)
        splits = []
        start = 0
        for idx in range(num_splits):
            chunk_size = base + (1 if idx < remainder else 0)
            end = start + chunk_size
            splits.append(items[start:end])
            start = end

        return splits
