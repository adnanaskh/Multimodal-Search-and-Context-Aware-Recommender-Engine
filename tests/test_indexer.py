import tempfile
from src.indexer import spimi_index, save_index, load_index, boolean_search


def test_spimi_and_search():
    docs = {
        "1": "Apple banana orange",
        "2": "Banana carrot",
        "3": "Apple carrot banana",
    }
    idx = spimi_index(docs, block_size=2, temp_dir=tempfile.mkdtemp())
    assert "apple" in idx
    assert "banana" in idx
    res = boolean_search(idx, "apple banana")
    assert set(res) == {"1", "3"}


def test_save_and_load():
    docs = {"1": "foo bar", "2": "bar baz"}
    idx = spimi_index(docs, block_size=2, temp_dir=tempfile.mkdtemp())
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    tf.close()
    save_index(idx, tf.name)
    loaded = load_index(tf.name)
    assert loaded == idx
