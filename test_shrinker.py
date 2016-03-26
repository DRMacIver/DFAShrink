from hypothesis import given, strategies as st, assume, example, note, settings
from shrinker import Shrinker, shrink
from hashlib import sha1
import json


@given(st.lists(st.binary()), st.booleans())
def test_consider_lang(lang, b):
    search = Shrinker(lambda x: (x in lang) ^ b)
    for l in lang:
        search.check(l)
        assert not search.check(l)

    for l in lang:
        state = 0
        for c in l:
            state = search.transition(state, c)
        assert search.accepting(state) ^ b


@example(1, b'\x01')
@example(0, b'')
@given(st.integers(0, 10), st.binary(average_size=20))
def test_shrink_length_language(n, b):
    assume(len(b) >= n)
    best = shrink(b, lambda x: len(x) >= n)
    assert best == bytes(n)


@given(st.binary())
@settings(max_examples=20)
def test_shrink_messy(s):
    b = sha1(s).digest()[0]
    search = Shrinker(lambda x: sha1(x).digest()[0] == b)
    search.check(s)
    search.shrink()
    assert sha1(search.best).digest()[0] == b
    candidates = [b''] + [bytes([i]) for i in range(256)] + [
        bytes([i, j]) for i in range(256) for j in range(256)
    ]
    note(repr(search))
    for c in candidates:
        if sha1(c).digest()[0] == b:
            actual_best = c
            break
    if actual_best != search.best:
        assert search.check(actual_best)
        search.shrink()
        assert search.best == actual_best


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
    search = Shrinker(is_valid_json)
    search.check(string)
    search.shrink()
    assert search.best == b"[]"


@given(st.recursive(
    st.lists(st.integers()), lambda s: st.lists(s, average_size=2)))
def test_shrink_longer_json(js):
    string = json.dumps(js).encode('utf-8')
    assert is_valid_json(string)
    assume(len(string) > 20)
    search = Shrinker(
        lambda s: len(s) > 2 and is_valid_json(s),
    )
    search.check(string)
    search.shrink()
    assert len(search.best) == 3
    assert search.best[0] == ord(b"[")
    assert search.best[2] == ord(b"]")
