import collections.abc
from typing import *

import pytest

from goats.common import iterables


def test_separable():
    """Test the type that defines a separable collection."""
    separables = [[1, 2], (1, 2), range(1, 3)]
    for arg in separables:
        assert isinstance(arg, iterables.Separable)
        assert iterables.Separable(arg) == arg
    nonseparables = ['a', '1, 2', 1, 1.0, slice(None), slice(1)]
    for arg in nonseparables:
        assert not isinstance(arg, iterables.Separable)
    value = 2
    separable = iterables.Separable(value)
    assert len(separable) == 1
    assert value in separable
    assert list(separable) == [2]
    values = [1, 2]
    separable = iterables.Separable(values)
    assert len(separable) == len(values)
    assert all(value in separable for value in values)
    assert list(separable) == list(values)
    separables = [
        iterables.Separable(None),
        iterables.Separable([]),
        iterables.Separable(()),
    ]
    for separable in separables:
        assert len(separable) == 0
        assert not list(separable)
    string = '1, 2'
    separable = iterables.Separable(string)
    assert len(separable) == 1
    assert string in separable
    assert string != separable
    assert list(separable) == [string]


def test_unique():
    """Test the function that searches a container for a unique option."""
    assert iterables.unique('a b c', ['a', 'd']) == 'a'
    assert iterables.unique('a b c', ['a', 'b']) is None


def test_unwrap():
    """Test the function that removes certain outer sequence types."""
    cases = [[3], (3,), [[3]], [(3,)], ([3],), ((3,),)]
    for case in cases:
        assert iterables.unwrap(case) == 3


def test_naked():
    """Test the function that identifies objects not wrapped in an iterable."""
    cases = [..., 'this', 0, 0.0, range(1), slice(None)]
    for case in cases:
        assert iterables.naked(case)
    cases = [(case,) for case in cases] + [(0,), [0]]
    for case in cases:
        assert not iterables.naked(case)


def test_missing():
    """Test the function that excludes 0 from truthiness evaluation."""
    assert iterables.missing(None)
    assert iterables.missing([])
    assert iterables.missing(())
    assert not iterables.missing(0)


def test_collection_mixin():
    """Test the mixin class that provides `Collection` methods."""
    class C0(iterables.CollectionMixin, collections.abc.Collection):
        def __init__(self, user: Iterable) -> None:
            self.user = user
            self.collect('user')

    class C1(iterables.CollectionMixin, collections.abc.Collection):
        def __init__(self) -> None:
            self.collect('user')
        def update(self, user: Iterable):
            self.user = user

    user = ['a', 1, '1']
    c = C0(user)
    assert len(c) == 3
    assert 'a' in c
    assert 'A' not in c
    assert list(c) == user
    c = C1()
    assert len(c) == 0
    assert list(c) == []
    c.update(user)
    assert 'a' in c
    assert 'A' not in c
    assert list(c) == user


def test_mapping_base():
    """Test the object that serves as a basis for concrete mappings."""
    class Incomplete(iterables.MappingBase):
        def __init__(self, mapping: Mapping) -> None:
            __mapping = mapping or {}
            super().__init__(__mapping.keys())

    class Implemented(iterables.MappingBase):
        def __init__(self, mapping: Mapping) -> None:
            __mapping = mapping or {}
            super().__init__(__mapping.keys())
            self.__mapping = __mapping
        def __getitem__(self, k: Any):
            if k in self.__mapping:
                return self.__mapping[k]
            raise KeyError(k)

    with pytest.raises(TypeError):
        c = Incomplete()
    in_dict = {'a': 1, 'b': 2}
    mapping = Implemented(in_dict)
    assert len(mapping) == len(in_dict)
    for key in in_dict.keys():
        assert key in mapping
    assert sorted(mapping) == sorted(in_dict)


def test_bijection():
    """Test the class the represents a bijective (one-to-one) mapping."""
    cases = [
        {
            'a': 1,
            'b': 2,
            'c': 3,
        },
        [
            ('a', 1),
            ('b', 2),
            ('c', 3),
        ],
    ]
    for case in cases:
        forward = iterables.Bijection(case)
        assert forward['a'] == 1
        assert forward['b'] == 2
        assert forward['c'] == 3
        inverse = forward.invert()
        assert inverse[1] == 'a'
        assert inverse[2] == 'b'
        assert inverse[3] == 'c'


def test_bijection_errors():
    test = {
        'a': 1,
        'b': 2,
        'c': 3,
    }
    invalid = {
        **test,
        'd': test['a'],
    }
    with pytest.raises(iterables.InjectiveTypeError):
        iterables.Bijection(invalid)


def test_object_registry():
    """Test the class that holds objects with metadata."""
    registry = iterables.ObjectRegistry({'this': [2, 3]})
    assert registry['this']['object'] == [2, 3]
    @registry.register(name='func', color='blue')
    def func():
        pass
    assert registry['func']['object'] == func
    assert registry['func']['color'] == 'blue'
    assert len(registry) == 2
    assert 'func' in registry
    assert sorted(registry) == sorted(['this', 'func'])


@pytest.fixture
def standard_entries():
    """A collection of well-behaved entries for a Table instance."""
    return [
        {'name': 'Gary', 'nickname': 'Gare-bear', 'species': 'cat'},
        {'name': 'Pickles', 'nickname': 'Pick', 'species': 'cat'},
        {'name': 'Ramon', 'nickname': 'Ro-ro', 'species': 'dog'},
        {'name': 'Frances', 'nickname': 'Frank', 'species': 'turtle'},
    ]


@pytest.fixture
def extra_key():
    """A collection of entries in which one has an extra key."""
    return [
        {'lower': 'a', 'upper': 'A'},
        {'lower': 'b', 'upper': 'B'},
        {'lower': 'c', 'upper': 'C', 'example': 'car'},
    ]


def test_table_lookup(standard_entries: list):
    """Test the object that supports multi-key look-up."""
    table = iterables.Table(standard_entries)
    gary = table(name='Gary')
    assert gary['nickname'] == 'Gare-bear'
    assert gary['species'] == 'cat'
    this = table(nickname='Ro-ro')
    assert this['name'] == 'Ramon'
    assert this['species'] == 'dog'
    with pytest.raises(iterables.TableLookupError):
        table(name='Simone')
    with pytest.raises(iterables.AmbiguousRequestError):
        table(species='cat')
    okay = table(species='cat', name='Pickles')
    assert okay['nickname'] == 'Pick'
    with pytest.raises(iterables.TableLookupError):
        table(species='dog', name='Gary', strict=True)
    with pytest.raises(iterables.TableLookupError):
        table(nickname='Yrag', species='dog', name='Gary', strict=True)


def test_table_errors(
    standard_entries: list,
    extra_key: list,
) -> None:
    """Regression test for `Table` error messages.

    This is separate from other tests in case we want to assert that `Table`
    raised a particular exception but we don't care what the actual message is.
    """
    standard = iterables.Table(standard_entries)
    extra = iterables.Table(extra_key)

    message = "Table has no common key 'example'"
    with pytest.raises(iterables.TableKeyError, match=message):
        standard(example='bird')
    with pytest.raises(iterables.TableKeyError, match=message):
        standard(example='bird', strict=True)
    with pytest.raises(iterables.TableKeyError, match=message):
        extra(example='car')
    message = "Table has no entry with species=dog and name=Gary"
    with pytest.raises(iterables.TableLookupError, match=message):
        standard(species='dog', name='Gary', strict=True)
    message = (
        "Table has no entry with"
        " nickname=Yrag, species=cat, and name=Gary"
    )
    with pytest.raises(iterables.TableLookupError, match=message):
        standard(nickname='Yrag', species='cat', name='Gary', strict=True)
    message = "Table has no entry with name=Simone"
    with pytest.raises(iterables.TableLookupError, match=message):
        standard(name='Simone')
    message = "The search criterion 'species=cat' is ambiguous"
    with pytest.raises(iterables.AmbiguousRequestError, match=message):
        standard(species='cat')


def test_table_modes(standard_entries: list):
    """Test the search modes available to Table."""
    def permute(d: dict, n: int=0) -> dict:
        """Permute the dict by `n` (anti-cyclic for n < 0)."""
        if n == 0:
            return d
        keys = list(d.keys())
        if n < 0:
            l = len(keys)
            perm = keys[l+n::-1] + keys[:l+n:-1]
        else:
            perm = keys[n:] + keys[:n]
        return {k: d[k] for k in perm}

    table = iterables.Table(standard_entries)
    valid = {'name': 'Gary', 'nickname': 'Gare-bear', 'species': 'cat'}
    permutations = []
    length = len(valid)
    for n in range(length):
        permutations.extend([permute(valid, n), permute(valid, n-length)])
    for request in permutations:
        entry = table(**request)
        assert entry['name'] == 'Gary'
        assert entry['nickname'] == 'Gare-bear'
        assert entry['species'] == 'cat'
    for request in permutations[:4]:
        entry = table(**{**request, **{'species': 'dog'}})
        assert entry['name'] == 'Gary'
        assert entry['nickname'] == 'Gare-bear'
        assert entry['species'] == 'cat'
    for request in permutations[4:]:
        entry = table(**{**request, **{'species': 'dog'}})
        assert entry['name'] == 'Ramon'
        assert entry['nickname'] == 'Ro-ro'
        assert entry['species'] == 'dog'
    with pytest.raises(iterables.TableLookupError):
        table(name='Gary', nickname='Gare-bear', species='dog', strict=True)


def test_table_getitem(extra_key: list):
    """Make sure we can get values of a common key via [] syntax."""
    table = iterables.Table(extra_key)
    subset = [entry['lower'] for entry in extra_key]
    assert table['lower'] == tuple(subset)
    with pytest.raises(iterables.TableKeyError):
        table['example']


def test_table_get(extra_key: list):
    """Make sure we can get value of any key, or a default value."""
    table = iterables.Table(extra_key)
    subset = [entry.get('example') for entry in extra_key]
    assert table.get('example') == tuple(subset)
    subset = [entry.get('example', -1) for entry in extra_key]
    assert table.get('example', -1) == tuple(subset)


def test_table_find(standard_entries: list):
    """Test table look-up by value."""
    table = iterables.Table(standard_entries)
    expected = {'name': 'Pickles', 'nickname': 'Pick', 'species': 'cat'}
    assert table.find('Pickles') == [expected]
    assert table.find('Pickles', unique=True) == expected
    expected = [
        {'name': 'Gary', 'nickname': 'Gare-bear', 'species': 'cat'},
        {'name': 'Pickles', 'nickname': 'Pick', 'species': 'cat'},
    ]
    assert table.find('cat') == expected
    with pytest.raises(iterables.AmbiguousValueError):
        table.find('cat', unique=True)
    with pytest.raises(iterables.MissingValueError):
        table.find('pidgeon')


def test_nothing():
    """Test the object that represents nothing."""
    assert not iterables.Nothing
    assert len(iterables.Nothing) == 0
    assert iterables.Nothing['at all'] is None
    assert iterables.Nothing(to_see='here') is None
    assert 'something' not in iterables.Nothing
    for _ in iterables.Nothing:
        assert False
    with pytest.raises(StopIteration):
        next(iterables.Nothing)
    this = iterables.NothingType()
    assert this is iterables.Nothing


def test_distribute():
    """Test the function that distributes one object over another."""
    expected = [('a', 1), ('a', 2), ('b', 1), ('b', 2)]
    assert list(iterables.distribute(['a', 'b'], [1, 2])) == expected
    expected = [('a', 1), ('a', 2)]
    assert list(iterables.distribute('a', [1, 2])) == expected
    expected = [('a', 1), ('b', 1)]
    assert list(iterables.distribute(['a', 'b'], 1)) == expected


def test_batch_replace():
    """Test replacing multiple characters in a string."""
    these = {'a': 'A', 'c': 'C'}
    string = 'abcd'
    expected = 'AbCd'
    assert iterables.batch_replace(string, these) == expected
