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
        assert list(separable) == []
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


def test_binary_groups():
    """Test the class that splits items into included and excluded."""
    groups = iterables.BinaryGroups(range(10), range(0, 10, 2))
    assert groups.included == (0, 2, 4, 6, 8)
    assert groups.excluded == (1, 3, 5, 7, 9)
    groups.sorter([2, 5])
    assert groups.included == (2, 5)
    assert groups.excluded == (0, 1, 3, 4, 6, 7, 8, 9)
    groups.sorter([2, 15])
    assert groups.included == (2,)
    assert groups.excluded == (0, 1, 3, 4, 5, 6, 7, 8, 9)
    items = {k: v for k, v in zip(['a', 'b', 'c', 'd'], range(4))}
    groups = iterables.BinaryGroups(items, ['a', 'd'])
    assert groups.included == {'a': 0, 'd': 3}
    assert groups.excluded == {'b': 1, 'c': 2}


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


def test_aliased_key():
    """Test the object that represents aliased mapping keys."""
    assert len(iterables.AliasedKey('t0')) == 1
    assert len(iterables.AliasedKey(('t0', 't1', 't2'))) == 3
    assert len(iterables.AliasedKey(['t0', 't1', 't2'])) == 3
    assert len(iterables.AliasedKey({'t0', 't1', 't2'})) == 3
    assert len(iterables.AliasedKey('t0', 't1', 't2')) == 3
    key = iterables.AliasedKey('t0', 't1', 't2')
    assert key + 't3' == iterables.AliasedKey('t0', 't1', 't2', 't3')
    assert key - 't2' == iterables.AliasedKey('t0', 't1')
    assert iterables.AliasedKey('a', 'b') == iterables.AliasedKey('b', 'a')


def test_alias_map():
    """Test the collection that groups aliases."""
    original = [('a', 'A'), 'b', ['c', 'C']]
    aliased = iterables.AliasMap(original)
    assert aliased['a'] == iterables.AliasedKey('a', 'A')
    assert aliased['A'] == iterables.AliasedKey('a', 'A')
    assert aliased['b'] == iterables.AliasedKey('b')
    assert aliased['c'] == iterables.AliasedKey('c', 'C')
    assert aliased['C'] == iterables.AliasedKey('c', 'C')


def test_aliased_mapping():
    """Test the object that represents a mapping with aliased keys."""
    # Set up mappings.
    _standard = {'this': 1, 'that': 2, 'the other': 3}
    _aliased = {
        ('this', 'first'): 1,
        ('that', 'second'): 2,
        ('the other', 'third'): 3,
    }
    _mixed = {
        'this': 1,
        ('that', 'second'): 2,
        ('the other',): 3,
    }
    standard = iterables.AliasedMapping(_standard)
    aliased = iterables.AliasedMapping(_aliased)
    mixed = iterables.AliasedMapping(_mixed)

    # Use the common keys to check values.
    for key, value in standard.items():
        assert value == aliased[key]
        assert value == mixed[key]

    # Check values by using aliases where they exist.
    for keys in _aliased:
        assert standard[keys[0]] == aliased[keys[1]]
        assert keys[1] not in standard
        assert mixed[keys[0]] == aliased[keys[1]]

    # Test aliased-key look-up.
    for key, value in aliased.items().aliased:
        assert aliased[key] == value

    # Containment checks should operate on the string keys.
    assert 'the other' in mixed and ('the other',) not in mixed

    # Check lengths of keys, values, and items.
    for mapping, n_keys in zip([standard, aliased, mixed], [3, 6, 4]):
        _check_aliased_keys(mapping, n_keys)

    # Key lists should be flat lists of strings.
    assert sorted(standard) == sorted(['this', 'that', 'the other'])
    assert sorted(aliased) == sorted([
        'this', 'first', 'that', 'second', 'the other', 'third'
    ])
    assert sorted(mixed) == sorted(['this', 'that', 'second', 'the other'])

    # The caller should be able to get the de-aliased mapping.
    dealiased = {
        'this': 1,
        'first': 1,
        'that': 2,
        'second': 2,
        'the other': 3,
        'third': 3,
    }
    assert aliased.flat == dealiased

    # The caller should be able to get known aliases but not set them.
    assert mixed.alias('that') == ('second',)
    assert mixed.alias('that', include=True) == ('that', 'second')
    with pytest.raises(TypeError):
        mixed.alias(this='THIS')


def _check_aliased_keys(mapping: iterables.AliasedMapping, n_keys: int):
    """Helper function for `test_aliased_mapping`."""
    assert len(mapping) == n_keys
    assert len(mapping.keys()) == n_keys
    assert len(mapping.values()) == n_keys
    assert len(mapping.items()) == n_keys
    assert len(list(mapping.keys().aliased)) == 3
    assert len(list(mapping.values().aliased)) == 3
    assert len(list(mapping.items().aliased)) == 3


def test_aliased_mutable_mapping():
    """Test the mutable version of an aliased mapping."""
    # Set up mappings.
    _aliased = {
        ('this', 'first'): 1,
        ('that', 'second'): 2,
        ('the other', 'third'): 3,
    }
    _mixed = {
        'this': 1,
        ('that', 'second'): 2,
        ('the other',): 3,
    }
    aliased = iterables.AliasedMutableMapping(_aliased)
    mixed = iterables.AliasedMutableMapping(_mixed)

    # Item assignment and updates should apply to all aliases.
    aliased['this'] = -10
    assert aliased['first'] == -10
    aliased.update({'this': 2, 'second': '2nd', 'new': -9.9})
    for key in ('this', 'first'):
        assert aliased[key] == 2
    for key in ('that', 'second'):
        assert aliased[key] == '2nd'
    assert aliased['new'] == -9.9

    # Assigning to a non-existant key should create a new item.
    with pytest.raises(KeyError):
        aliased['unused']
    aliased['unused'] = 1.234
    assert aliased['unused'] == 1.234

    # Removing an item by one alias should remove the value and all aliases.
    del aliased['this']
    for key in ('this', 'first'):
        assert key not in aliased

    # The caller should be able to register new aliases.
    mixed.alias(this='THIS')
    assert 'THIS' in mixed and mixed['this'] == mixed['THIS']
    with pytest.raises(TypeError):
        mixed.alias('this', that='new alias')
    with pytest.raises(ValueError):
        mixed.alias('this', 'that')

    # Attempting to assign an existing alias should be an error.
    with pytest.raises(KeyError):
        aliased.alias(that='third')

    # Assigning no aliases should leave the key unchanged.
    mixed.alias(this=())
    assert 'this' in mixed


def test_immutable_from_mutable():
    """Test creating an immutable aliased mapping from a mutable one."""
    _aliased = {
        ('this', 'first'): 1,
        ('that', 'second'): 2,
        ('the other', 'third'): 3,
    }
    mutable = iterables.AliasedMutableMapping(_aliased)
    immutable = iterables.AliasedMapping(mutable)
    assert isinstance(immutable, iterables.AliasedMapping)
    assert not hasattr(immutable, 'update')
    with pytest.raises(TypeError):
        immutable['this'] = -10
    with pytest.raises(TypeError):
        del immutable['this']


def test_aliased_mapping_idempotence():
    """Make sure we can create an aliased mapping from an aliased mapping."""
    user = {
        ('this', 'first'): 1,
        ('that', 'second'): 2,
        ('the other', 'third'): 3,
    }
    aliased = iterables.AliasedMapping(user)
    from_aliased = iterables.AliasedMapping(aliased)
    assert from_aliased == aliased
    for key in aliased:
        assert aliased.alias(key) == from_aliased.alias(key)
    from_aliased = iterables.AliasedMapping.of(aliased)
    assert from_aliased == aliased
    from_aliased = iterables.AliasedMapping.fromkeys(aliased)
    assert from_aliased.keys() == aliased.keys()


def test_aliased_mapping_of():
    """Test the class method that creates an aliased mapping of a mapping."""
    this = {
        'a': {'aliases': ('A', 'a0'), 'name': 'Annabez', 'k': ['Ka']},
        'b': {'aliases': 'B', 'name': 'Borb', 'k': ('Kb', 'KB')},
        'C': {'aliases': ('c',), 'name': 'Chrunk'}
    }
    aliased = iterables.AliasedMapping.of(this)
    expected = {
        'a': {'name': 'Annabez', 'k': ['Ka']},
        'A': {'name': 'Annabez', 'k': ['Ka']},
        'a0': {'name': 'Annabez', 'k': ['Ka']},
        'b': {'name': 'Borb', 'k': ('Kb', 'KB')},
        'B': {'name': 'Borb', 'k': ('Kb', 'KB')},
        'C': {'name': 'Chrunk'},
        'c': {'name': 'Chrunk'},
    }
    assert aliased.flat == expected
    aliased = iterables.AliasedMapping.of(this, value_key='name')
    expected = {
        'a': 'Annabez',
        'A': 'Annabez',
        'a0': 'Annabez',
        'b': 'Borb',
        'B': 'Borb',
        'C': 'Chrunk',
        'c': 'Chrunk',
    }
    assert aliased.flat == expected
    aliased = iterables.AliasedMapping.of(this, alias_key='k', value_key='name')
    expected = {
        'a': 'Annabez',
        'Ka': 'Annabez',
        'b': 'Borb',
        'Kb': 'Borb',
        'KB': 'Borb',
        'C': 'Chrunk',
    }
    assert aliased.flat == expected


def test_aliased_mapping_fromkeys():
    """Test the class method that creates an aliased mapping from keys."""
    this = {
        'a': {'aliases': ('A', 'a0'), 'name': 'Annabez', 'k': ['Ka']},
        'b': {'aliases': 'B', 'name': 'Borb', 'k': ('Kb', 'KB')},
        'C': {'aliases': ('c',), 'name': 'Chrunk'}
    }
    aliased = iterables.AliasedMapping.fromkeys(this)
    expected = {
        'a': None,
        'A': None,
        'a0': None,
        'b': None,
        'B': None,
        'C': None,
        'c': None,
    }
    assert aliased.flat == expected
    aliased = iterables.AliasedMapping.fromkeys(this, value=-4.5)
    expected = {
        'a': -4.5,
        'A': -4.5,
        'a0': -4.5,
        'b': -4.5,
        'B': -4.5,
        'C': -4.5,
        'c': -4.5,
    }
    assert aliased.flat == expected


def test_aliased_mapping_extract_keys():
    """Test the class method that extracts aliased keys."""
    this = {
        'a': {'aliases': ('A', 'a0'), 'name': 'Annabez', 'k': ['Ka']},
        'b': {'aliases': 'B', 'name': 'Borb', 'k': ('Kb', 'KB')},
        'C': {'aliases': ('c',), 'name': 'Chrunk'}
    }
    keys = iterables.AliasedMapping.extract_keys(this)
    expected = [
        ['a', 'A', 'a0'],
        ['b', 'B'],
        ['C', 'c'],
    ]
    assert keys == [iterables.AliasedKey(k) for k in expected]


def test_aliased_keysview():
    """Test the custom keys view for aliased mappings."""
    d1 = {
        ('this', 'first'): 1,
        ('that', 'second'): 2,
        ('the other', 'third'): 3,
    }
    d2 = d1.copy()
    d2[('that', 'second')] = -20
    assert d1 != d2
    a1 = iterables.AliasedMapping(d1)
    a2 = iterables.AliasedMapping(d2)
    assert a1 != a2
    assert a1.keys() == a2.keys()
    assert a1.keys() == d1.keys()


def test_aliased_mapping_copy():
    """Test the copy method of an aliased mapping."""
    d1 = {
        ('this', 'first'): 1,
        ('that', 'second'): 2,
        ('the other', 'third'): 3,
    }
    a1 = iterables.AliasedMapping(d1)
    a2 = a1.copy()
    assert a1 == a2
    assert a1 is not a2
    assert a1.keys() == a2.keys()
    assert a1.keys() == d1.keys()


def test_aliased_mapping_merge():
    """Test the merge operator on aliased mappings."""
    d1 = {
        ('this', 'first'): 1,
        ('that', 'second'): 2,
    }
    d2 = {
        ('the other', 'third'): 3,
        ('another', 'fourth'): 4,
    }
    a1 = iterables.AliasedMapping(d1)
    a2 = iterables.AliasedMapping(d2)
    merged = a1 | a2
    expected = sorted(list(a1.keys()) + list(a2.keys()))
    assert sorted(merged.keys()) == expected
    expected = sorted(list(a1.values()) + list(a2.values()))
    assert sorted(merged.values()) == expected
    assert merged != a1
    assert merged != a2


def test_namemap():
    """Test the object that maps aliases to names."""
    references = [
        ['a', 'b', 'c'],
        {'a': 1, 'b': 2, 'c': 3},
    ]
    cases = [
        # Iterable of aliased keys
        [['b', 'B'], ['c', 'c0', 'C']],
        # Mapping from name to aliases
        {'b': 'B', 'c': ['c0', 'C']},
        # Mapping from name to mapping with default alias key
        {'b': {'aliases': 'B'}, 'c': {'aliases': ['c0', 'C']}},
    ]
    n_aliases = 6 # Non-trivial to compute for an arbitrary case in cases
    for aliases in cases:
        if isinstance(aliases, Mapping):
            namemap = iterables.NameMap(aliases)
            check_namemap_defs_only(namemap)
        else:
            with pytest.raises(TypeError):
                namemap = iterables.NameMap(aliases)
        for names in references:
            namemap = iterables.NameMap(aliases, refs=names)
            check_namemap_with_refs(namemap, n_aliases, names)


def check_namemap_defs_only(namemap):
    """Helper for `test_namemap` without given `refs`."""
    for alias in ['b', 'B']:
        assert namemap[alias] == 'b'
    for alias in ['c', 'C', 'c0']:
        assert namemap[alias] == 'c'


def check_namemap_with_refs(namemap, n_aliases, names):
    """Helper for `test_namemap` with given `refs`."""
    check_namemap_defs_only(namemap)
    assert namemap['a'] == 'a'
    assert len(namemap) == n_aliases
    assert all(name in namemap for name in names)


def test_namemap_copy():
    """Test the copy method of the alias -> name mapping."""
    names = ['a', 'b', 'c']
    aliases = {'b': {'alt': 'B'}, 'c': {'alt': ['c0', 'C']}}
    namemap = iterables.NameMap(aliases, names, key='alt')
    copied = namemap.copy()
    assert copied.keys() == namemap.keys()
    assert copied.values() == namemap.values()
    assert copied is not namemap


def test_namemap_key():
    """Test the alias -> name mapping with non-default alias key"""
    names = ['a', 'b', 'c']
    aliases = {'b': {'alt': 'B'}, 'c': {'alt': ['c0', 'C']}}
    namemap = iterables.NameMap(aliases, names, key='alt')
    assert namemap['a'] == 'a'
    for alias in ['b', 'B']:
        assert namemap[alias] == 'b'
    for alias in ['c', 'C', 'c0']:
        assert namemap[alias] == 'c'


def test_namemap_invert():
    """Test the ability to invert an aliases -> name mapping."""
    names = ['a', 'b', 'c']
    aliases = [['b', 'B'], ['c', 'c0', 'C']]
    namemap = iterables.NameMap(aliases, names).invert()
    assert sorted(namemap['a']) == sorted(['a'])
    assert sorted(namemap['b']) == sorted(['b', 'B'])
    assert sorted(namemap['c']) == sorted(['c', 'c0', 'C'])


def test_aliased_cache():
    """Test the object that represents a cache with aliased look-up."""
    mapping = {'a': 1, 'b': 2, 'c': 3}
    cases = [
        [['b', 'B'], ['c', 'c0', 'C']], # Iterable of aliased keys
        {'b': 'B', 'c': ['c0', 'C']}, # Mapping from name to aliases
    ]
    for aliases in cases:
        cache = iterables.AliasedCache(mapping, aliases)
        assert all(v is None for v in cache.values())
        assert cache['a'] == 1
        assert cache['b'] == 2
        assert cache['B'] is cache['b']
        assert cache['C'] == 3
        assert list(cache.values().aliased) == list(mapping.values())
    aliases = ['b', 'B', 'c', 'c0', 'C'] # Flat iterable of keys -> no aliases
    cache = iterables.AliasedCache(mapping, aliases)
    assert all(v is None for v in cache.values())
    assert cache['a'] == 1
    assert cache['b'] == 2
    for key in set(aliases) - set(mapping.keys()):
        with pytest.raises(iterables.AliasedKeyError):
            cache[key]


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
