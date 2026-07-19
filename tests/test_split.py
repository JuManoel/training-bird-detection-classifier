"""Stratified split smoke tests."""

from pipelines.shared.split import stratified_group_split, stratified_split


def test_stratified_split_no_empty_val_for_large_class():
    items = list(range(125))
    labels = ["A"] * 125
    split = stratified_split(items, labels, train_ratio=0.8, seed=42)
    assert len(split.train) == 100
    assert len(split.val) == 25


def test_stratified_group_split_keeps_siblings_together():
    # Two crops from photo g1, two from g2, two from g3 — all species A
    items = ["g1a", "g1b", "g2a", "g2b", "g3a", "g3b"]
    labels = ["A"] * 6
    groups = ["g1", "g1", "g2", "g2", "g3", "g3"]
    split = stratified_group_split(items, labels, groups, train_ratio=0.8, seed=0)
    for sibling_group in ("g1", "g2", "g3"):
        members = [i for i in items if i.startswith(sibling_group)]
        in_train = sum(1 for m in members if m in split.train)
        in_val = sum(1 for m in members if m in split.val)
        assert in_train in (0, 2)
        assert in_val in (0, 2)
        assert in_train + in_val == 2
