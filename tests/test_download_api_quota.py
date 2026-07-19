"""API quota splitting and download helper tests."""

from pipelines.download_data.application.download import _split_api_quotas


def test_split_api_quotas_both_enabled():
    assert _split_api_quotas(10, True, True) == (5, 5)
    assert _split_api_quotas(11, True, True) == (6, 5)


def test_split_api_quotas_single_source():
    assert _split_api_quotas(7, True, False) == (7, 0)
    assert _split_api_quotas(7, False, True) == (0, 7)
    assert _split_api_quotas(7, False, False) == (0, 0)


def test_split_api_quotas_zero():
    assert _split_api_quotas(0, True, True) == (0, 0)
