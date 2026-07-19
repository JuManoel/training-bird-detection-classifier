"""Stratified split smoke tests."""

from pipelines.shared.split import stratified_split


def test_stratified_split_no_empty_val_for_large_class():
    items = list(range(125))
    labels = ["A"] * 125
    split = stratified_split(items, labels, train_ratio=0.8, seed=42)
    assert len(split.train) == 100
    assert len(split.val) == 25
