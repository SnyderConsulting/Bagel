import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:  # pragma: no cover - optional torch dependency for tests
    import torch  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    import types

    torch = types.ModuleType("torch")
    utils_module = types.ModuleType("torch.utils")
    data_module = types.ModuleType("torch.utils.data")

    class _IterableDataset:  # minimal stub for testing
        pass

    def _get_worker_info():
        return None

    data_module.IterableDataset = _IterableDataset
    data_module.get_worker_info = _get_worker_info
    utils_module.data = data_module
    torch.utils = utils_module

    sys.modules.setdefault("torch", torch)
    sys.modules.setdefault("torch.utils", utils_module)
    sys.modules.setdefault("torch.utils.data", data_module)

from data.distributed_iterable_dataset import DistributedIterableDataset


class DummyLoopDataset(DistributedIterableDataset):
    def __init__(self, data_paths, dataset_name="dummy", local_rank=0, world_size=1, num_workers=1, max_loops=2):
        super().__init__(dataset_name, local_rank=local_rank, world_size=world_size, num_workers=num_workers)
        self.max_loops = max_loops
        self.data_paths = list(data_paths)
        self.set_epoch()

    def __iter__(self):
        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        loops = 0
        while loops < self.max_loops:
            for path in data_paths_per_worker:
                yield (worker_id, path)
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")
            loops += 1


class DistributedIterableDatasetTest(unittest.TestCase):
    def test_ranks_and_workers_receive_data(self):
        data_paths = ["file0"]
        world_size = 4
        num_workers = 3

        for local_rank in range(world_size):
            dataset = DummyLoopDataset(
                data_paths,
                dataset_name="regression",
                local_rank=local_rank,
                world_size=world_size,
                num_workers=num_workers,
            )
            dataset.set_epoch(seed=0)
            self.assertGreater(
                len(dataset.data_paths_per_rank),
                0,
                msg=f"Rank {local_rank} did not receive any data paths",
            )

            for worker_id in range(num_workers):
                worker_info = SimpleNamespace(id=worker_id, num_workers=num_workers)
                with patch("torch.utils.data.get_worker_info", return_value=worker_info):
                    data_paths_per_worker, reported_worker_id = dataset.get_data_paths_per_worker()

                self.assertEqual(worker_id, reported_worker_id)
                self.assertGreater(
                    len(data_paths_per_worker),
                    0,
                    msg=f"Worker {worker_id} on rank {local_rank} received no data paths",
                )

    def test_iterator_yields_before_repeat(self):
        data_paths = ["file0"]
        world_size = 4
        num_workers = 3
        dataset = DummyLoopDataset(
            data_paths,
            dataset_name="regression",
            local_rank=0,
            world_size=world_size,
            num_workers=num_workers,
            max_loops=2,
        )
        dataset.set_epoch(seed=123)

        worker_info = SimpleNamespace(id=1, num_workers=num_workers)
        with patch("torch.utils.data.get_worker_info", return_value=worker_info):
            with patch("builtins.print") as mock_print:
                iterator = iter(dataset)
                first_item = next(iterator)

        self.assertEqual(first_item, (1, "file0"))
        mock_print.assert_not_called()


if __name__ == "__main__":
    unittest.main()
