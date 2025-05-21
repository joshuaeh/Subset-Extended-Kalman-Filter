import pytest

def test_import_sekf():
    try:
        import sekf
    except ImportError:
        pytest.fail("Could not import the 'sekf' package")