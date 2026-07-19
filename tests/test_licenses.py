"""License filter tests."""

from pipelines.shared.licenses import is_license_allowed


def test_allow_cc_by():
    assert is_license_allowed("CC-BY-4.0")
    assert is_license_allowed("https://creativecommons.org/licenses/by/4.0/")


def test_allow_unknown_by_default():
    assert is_license_allowed(None)
    assert is_license_allowed("")


def test_deny_all_rights():
    assert not is_license_allowed("All Rights Reserved")


def test_deny_unknown_when_required():
    assert not is_license_allowed(None, allow_unknown=False)
