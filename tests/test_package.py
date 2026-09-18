import nkscape as nk


def test_version_is_exposed() -> None:
    assert nk.__version__ == "0.0.1"
