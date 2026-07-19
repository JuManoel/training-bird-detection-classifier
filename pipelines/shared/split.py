"""Stratified train/val split helpers."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class DatasetSplit(Generic[T]):
    train: list[T]
    val: list[T]


def stratified_split(
    items: list[T],
    labels: list[str],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> DatasetSplit[T]:
    """Split items 80/20 stratified by label. Tiny classes still get at least one train sample."""
    if len(items) != len(labels):
        raise ValueError("items and labels must have the same length")
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be in (0, 1)")

    rng = random.Random(seed)
    by_label: dict[str, list[T]] = defaultdict(list)
    for item, label in zip(items, labels, strict=True):
        by_label[label].append(item)

    train: list[T] = []
    val: list[T] = []
    for group in by_label.values():
        group = list(group)
        rng.shuffle(group)
        if len(group) == 1:
            train.extend(group)
            continue
        n_train = max(1, int(round(len(group) * train_ratio)))
        n_train = min(n_train, len(group) - 1)
        train.extend(group[:n_train])
        val.extend(group[n_train:])

    rng.shuffle(train)
    rng.shuffle(val)
    return DatasetSplit(train=train, val=val)


def stratified_group_split(
    items: list[T],
    labels: list[str],
    groups: list[str],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> DatasetSplit[T]:
    """Stratified split that keeps all items of the same group in one split.

    Used so multiple bird crops from one source photo stay together in train or val.
    """
    if not (len(items) == len(labels) == len(groups)):
        raise ValueError("items, labels, and groups must have the same length")
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be in (0, 1)")

    # One representative label per group (species of the source photo).
    group_label: dict[str, str] = {}
    group_items: dict[str, list[T]] = defaultdict(list)
    for item, label, group in zip(items, labels, groups, strict=True):
        group_items[group].append(item)
        group_label.setdefault(group, label)

    group_ids = list(group_items.keys())
    group_labels = [group_label[g] for g in group_ids]
    group_split = stratified_split(
        group_ids, group_labels, train_ratio=train_ratio, seed=seed
    )

    train: list[T] = []
    val: list[T] = []
    for gid in group_split.train:
        train.extend(group_items[gid])
    for gid in group_split.val:
        val.extend(group_items[gid])
    return DatasetSplit(train=train, val=val)


def early_stop_patience(epochs: int, fraction: float = 0.05) -> int:
    """Consecutive epochs without improvement = max(1, floor(fraction * epochs))."""
    return max(1, int(epochs * fraction))
