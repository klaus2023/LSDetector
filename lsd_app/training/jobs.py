"""Training job descriptors for the first engineering pass."""

from __future__ import annotations


def describe_training_jobs() -> list[str]:
    """Return the training jobs that will be promoted from legacy scripts."""

    return [
        "lsdsam-finetune-target",
        "lsdsam-domain-adaptation",
        "lsdformer-train-baseline",
    ]
