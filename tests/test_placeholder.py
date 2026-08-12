def test_import_src_package():
    import importlib
    src = importlib.import_module("src")
    assert getattr(src, "__file__", None) is not None
