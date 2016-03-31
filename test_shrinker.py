from hypothesis import given, strategies as st, assume, example, settings
from shrinker import shrink
from hashlib import sha1
import json
import os

settings.register_profile(
    'default', settings(max_examples=200)
)

settings.load_profile(os.getenv('HYPOTHESIS_PROFILE', 'default'))


@example(1, b'\x01')
@example(3, b'\x00\x00\x01')
@given(st.integers(0, 10), st.binary(average_size=20))
def test_shrink_length_language(n, b):
    assume(len(b) >= n)
    best = shrink(b, lambda x: len(x) >= n)
    assert best == bytes(n)


@given(st.binary())
def test_shrink_messy(s):
    b = sha1(s).digest()[0]
    shrunk = shrink(s, lambda x: sha1(x).digest()[0] == b)
    assert sha1(shrunk).digest()[0] == b


def is_valid_json(string):
    try:
        string = string.decode('utf-8')
    except UnicodeDecodeError:
        return False
    try:
        value = json.loads(string)
    except json.JSONDecodeError:
        return False
    try:
        iter(value)
    except TypeError:
        return False
    return True


@given(st.recursive(
    st.lists(st.integers()), lambda s: st.lists(s, average_size=2)))
def test_shrink_json(js):
    string = json.dumps(js).encode('utf-8')
    assert is_valid_json(string)
    shrunk = shrink(string, is_valid_json)
    assert shrunk == b"[]"


@example([1000000000000000000])
@given(st.recursive(
    st.lists(st.integers()), lambda s: st.lists(s, average_size=2)))
def test_shrink_longer_json(js):
    string = json.dumps(js).encode('utf-8')
    assert is_valid_json(string)
    assume(len(string) > 20)
    shrunk = shrink(string, lambda s: len(s) > 2 and is_valid_json(s))
    assert len(shrunk) == 3
    assert shrunk[0] == ord(b"[")
    assert shrunk[2] == ord(b"]")
