"""Utility helpers for Bagel inference."""

from .factory import (
    InferenceProcessors,
    build_model,
    build_processors,
    load_checkpoint,
    resolve_checkpoint_dir,
)
from .pipeline import glob_examples, predict_single_edit, run_batch

# Backwards compatibility: earlier versions exposed ``glob_pairs`` which
# returned (reference, input) tuples. The new ``glob_examples`` additionally
# surfaces optional ground-truth paths. Re-export ``glob_pairs`` so external
# callers expecting the old symbol continue to work.
def glob_pairs(dataset_root):
    """Compatibility wrapper for :func:`glob_examples`."""

    triples = glob_examples(dataset_root)
    return [(ref_path, input_path) for ref_path, input_path, _ in triples]

__all__ = [
    "InferenceProcessors",
    "build_model",
    "build_processors",
    "load_checkpoint",
    "resolve_checkpoint_dir",
    "predict_single_edit",
    "run_batch",
    "glob_examples",
    "glob_pairs",
]
