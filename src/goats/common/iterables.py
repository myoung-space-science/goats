import abc
import collections.abc
import inspect
from itertools import product
import functools
import numbers
from typing import *


def unique(target: Container, options: Iterable):
    """Search `target` for a unique element from `options`."""
    for option in options:
        others = set(options) - {option}
        no_others = all(this not in target for this in others)
        if option in target and no_others:
            return option


T = TypeVar('T')
def unwrap(obj: Union[T, Iterable]) -> Union[T, list, tuple]:
    """Remove redundant outer lists and tuples.

    This function will strip away enclosing instances of `list` or `tuple`, as
    long as they contain a single item, until it finds an object of a different
    type or an empty `list` or `tuple`.

    Parameters
    ----------
    obj : Any
        The object to "unwrap".

    Returns
    -------
    Any
        The element enclosed in multiple instances of `list` or `tuple`, or an
        empty `list` or `tuple`.

    Examples
    --------
    Unwrap numbers enclosed in increasingly deeper lists:

    >>> cases = [[1], [[2]], [[[3]]], [[[[4]]]]]
    >>> for case in cases:
    ...     print(iterables.unwrap(case))
    ... 
    1
    2
    3
    4

    It preserves numbers and strings that are already unwrapped:

    >>> iterables.unwrap(42)
    42
    >>> iterables.unwrap('string')
    'string'

    It works with multiple wrapped elements:

    >>> iterables.unwrap([1, 2])
    [1, 2]
    >>> iterables.unwrap([[1, 2]])
    [1, 2]
    >>> iterables.unwrap(['one', 'two'])
    ['one', 'two']
    >>> iterables.unwrap([['one', 'two']])
    ['one', 'two']

    It stops at an empty `list` or `tuple`:

    >>> iterables.unwrap([])
    []
    >>> iterables.unwrap(())
    ()
    >>> iterables.unwrap(list())
    []
    >>> iterables.unwrap(tuple())
    ()
    >>> iterables.unwrap([[]])
    []
    >>> iterables.unwrap([()])
    ()
    """
    seed = [obj]
    wrapped = (list, tuple)
    while isinstance(seed, wrapped) and len(seed) == 1:
        seed = seed[0]
    return seed


def get_nested_element(mapping: Mapping, levels: Iterable):
    """Walk a mapping to get the element at the last level."""
    this = mapping[levels[0]]
    if len(levels) > 1:
        for level in levels[1:]:
            this = this[level]
    return this


def missing(this: Any) -> bool:
    """True if `this` is null or empty.

    This function allows the user to programmatically test for objects that are
    logically ``False`` except for numbers equivalent to 0.
    """
    if isinstance(this, numbers.Number):
        return False
    return not bool(this)


def transpose_list(list_in: List[list]) -> List[list]:
    """Transpose a logically 2-D list.

    This function works by collecting the item at index `i` in each element of
    `list_in` in a list and storing that list in index `i` of the result.

    Parameters
    ----------
    list_in : list of lists

    The logically 2-D list to transpose. If `list_in` has effective dimensions N
    x M (i.e., it consists of N lists of length M), the result will be a list
    with effective dimensions M x N.

    Examples
    --------
    The following example illustrates how this function collects the 0th item
    in each element (e.g., 1 and 2) into the 0th element of the result, and so
    on.
    >>> list_in = [[1, 'a', 'A'], [2, 'b', 'B']]
    >>> transpose_list(list_in)
    [[1, 2], ['a', 'b'], ['A', 'B']]
    """
    element_length = len(list_in[0])
    return [
        [element[i] for element in list_in]
        for i in range(element_length)
    ]


def slice_to_range(s: slice, stop: int=0) -> range:
    """Attempt to convert a slice to a range."""
    if stop is None:
        raise TypeError(f"Cannot convert {s} to a range.")
    start = s.start or 0
    stop = s.stop or stop
    step = s.step or 1
    return range(start, stop, step)


def string_to_list(string: str) -> list:
    """Convert a string representation of a list to a list"""
    return [numeric_cast(i) for i in string.strip('[]').split(',')]


def numeric_cast(string: str) -> Union[int, float]:
    """Convert a string to an ``int`` or ``float``."""
    try:
        return float(string)
    except ValueError:
        return int(string)


def naked(targets: Any) -> bool:
    """True if the target objects are not wrapped in an iterable.

    This truth value can be useful in sitations when the user needs to parse
    arbitrary positional arguments (e.g., ``*args``) and wants to
    programmatically handle these cases.
    """
    singular = (int, float, str)
    return (
        targets is Ellipsis
        or
        isinstance(targets, (slice, range, *singular))
    )


def show_at_most(n: int, values: Iterable[Any], separator: str=',') -> str:
    """Create a string with at most `n` values."""
    seq = list(values)
    if len(seq) <= n:
        return separator.join(str(v) for v in seq)
    truncated = [*seq[:n-1], '...', seq[-1]]
    return separator.join(str(v) for v in truncated)


class ReprStrMixin:
    """A mixin class that provides support for `__repr__` and `__str__`."""

    def __repr__(self) -> str:
        """An unambiguous representation of this object."""
        return f"{self.__class__.__qualname__}({self})"


class RequiredAttrMeta(abc.ABCMeta):
    """A metaclass for requiring attributes to be defined in `__init__`.

    Based on https://stackoverflow.com/a/55571677/4739101. When used as a
    metaclass for a base class, this class will prevent users from creating new
    instances of a subclass without defining all required attributes listed in
    `RequiredAttrMeta._required`.
    """
    _required = []

    def __call__(self, *args, **kwargs):
        """Prevent instantiation without required attributes."""
        instance = super().__call__(*args, **kwargs)
        cls = self.__qualname__
        for attr in self._required:
            if not hasattr(instance, attr):
                raise NotImplementedError(
                    f"Can't instantiate {cls} without attribute '{attr}'"
                )
        return instance


# See MappingBase for a potentially more intuitive way to provide these features
# when implementing `collections.abc.Mapping`. This class still has the
# advantage of dynamically accessing the given collection to allow updates when
# implementing `collections.abc.MutableMapping`.
class CollectionMixin:
    """A mixin class for defining required `Collection` methods.

    Classes may use this as a mixin class by passing the name of the target
    collection to `collect`. This class supports updates to the target
    collection by always retrieving the current value. If the target collection
    is not an instance attribute (e.g., because it has not yet been defined),
    this class will use an empty iterable. Note that this behavior provides
    flexibility at the risk of a class appearing empty because the developer did
    not assign the target collection to an instance attribute.

    See Examples for a suggested pattern to use when programmatically defining
    concrete implementations of `collections.abc.Mapping`. Note that Python's
    MRO requires that this class come before the abstract class for which it
    provides methods.

    Examples
    --------
    Define a simple collection-based class::

        from typing import Iterable

        class MyCollection(CollectionMixin):
            def __init__(self, collection: Iterable):
                self.members = collection
                self.collect('members')

    Create an instance and confirm that it behaves correctly::

        myc = MyCollection(['me', 'myself', 'I'])
        print(len(myc))     # -> 3
        print(list(myc))    # -> ['me', 'myself', 'I']
        print('me' in myc)  # -> True
        print('you' in myc) # -> False

    Modify that class so that it requires the user to set the target collection
    after instantiation::

        class MyCollection(CollectionMixin):
            def __init__(self):
                self.collect('collection')

            def update(self, collection: Iterable):
                self.collection = collection

    Create a new instance and test the behavior before and after setting the
    target collection::

        myc = MyCollection()
        print(len(myc))     # -> 0
        print(list(myc))    # -> []
        print('me' in myc)  # -> False
        print('you' in myc) # -> False
        myc.update(['me', 'myself', 'I'])
        print(len(myc))     # -> 3
        print(list(myc))    # -> ['me', 'myself', 'I']
        print('me' in myc)  # -> True
        print('you' in myc) # -> False

    Define a custom mapping that forces all keys to lower case, overwriting in
    insertion order as necessary. This mapping implements
    `collections.abc.Mapping` but uses `CollectionMixin` to implement the
    abstract methods inherited from `collections.abc.Collection`::

        import collections.abc
        from typing import Dict, Any

        class CaseInsensitive(CollectionMixin, collections.abc.Mapping):
            def __init__(self, user: Dict[str, Any]) -> None:
                self.members = {k.lower(): v for k, v in user.items()}
                self.collect('members')

            def __getitem__(self, key: str):
                if key.lower() in self.members:
                    return self.members[key.lower()]
                if key.upper() in self.members:
                    return self.members[key.upper()]
                raise KeyError(f"No {key} in MyMapping")

    Create an instance, confirm that its length is only the number of unique
    case-insensitive keys, confirm that later values take priority, and that
    item look-up behaves as expected::

        mym = CaseInsensitive({'a': 10, 'b': 20, 'c': 30, 'B': -100})
        print(len(mym)) # -> 3
        print(mym['a']) # -> 10
        print(mym['A']) # -> 10
        print(mym['b']) # -> -100
        print(mym['d']) # -> KeyError: 'No d in MyMapping'

    """
    def collect(self, name: str):
        """Identify the target collection."""
        self._collection_name = name

    @property
    def _collection(self) -> Collection:
        """The base collection."""
        if hasattr(self, '_collection_name'):
            return getattr(self, self._collection_name, ())
        indent = ' ' * len('AttributeError: ')
        raise AttributeError(
            "Cannot collect unknown attribute."
            f"\n{indent}Please pass the name of the underlying collection"
            " via the collect method"
        ) from None

    def __contains__(self, key: str) -> bool:
        """True if `key` names a member of this collection."""
        return key in self._collection

    def __len__(self) -> int:
        """The number of members in this collection."""
        return len(self._collection)

    def __iter__(self) -> Iterator:
        """Iterate over members of this collection."""
        return iter(self._collection)


class MappingBase(collections.abc.Mapping):
    """A partial implementation of `collections.abc.Mapping`.

    This abstract base class is designed to serve as a basis for easily creating
    concrete implementations of `collections.abc.Mapping`. It defines simple
    implementations, based on a user-provided collection, for the abstract
    methods required by `collections.abc.Collection` (`__contains__`, `__len__`,
    and `__iter__`) but leaves `__getitem__` abstract.

    Examples
    --------
    The following class implements `collections.abc.Mapping`::

        class Implemented(MappingBase):

            def __init__(self, mapping: Mapping) -> None:
                __mapping = mapping or {}
                super().__init__(__mapping.keys())
                self.__mapping = __mapping

            def __getitem__(self, k: Any):
                if k in self.__mapping:
                    return self.__mapping
                raise KeyError(k)

    A trivial example demonstrates its behavior::

    >>> in_dict = {'a': 1, 'b': 2}
    >>> mapping = Implemented(in_dict)
    >>> len(mapping)
    2
    >>> list(mapping.keys())
    ['a', 'b']
    >>> list(mapping.values())
    [1, 2]

    Attemps to instantiate the following class will raise a `TypeError` because
    it doesn't define `__getitem__`::

        class Incomplete(MappingBase):

            def __init__(self, mapping: Mapping) -> None:
                __mapping = mapping or {}
                super().__init__(__mapping.keys())

    """

    def __init__(self, __collection: Collection) -> None:
        """Initialize this instance with the base collection.

        Parameters
        ----------
        __collection
            Any concrete implementation of `collections.abc.Collection`. This
            attribute's implementations of the required collection methods will
            support the equivalent implementations for this mapping.
        """
        self.__collection = __collection

    def __contains__(self, key: str) -> bool:
        """True if `key` names a member of this collection."""
        return key in self.__collection

    def __len__(self) -> int:
        """The number of members in this collection."""
        return len(self.__collection)

    def __iter__(self) -> Iterator:
        """Iterate over members of this collection."""
        return iter(self.__collection)


# This class is kind of awkward to use. It is only here because
# `base.DataMapping` currently uses it.
class AliasesMixin:
    """An abstract mixin class to support aliased key look-up.

    Classes may use this as a mixin class by defining `names` and `aliases`
    properties. The `names` property should return a list of canonical names in
    the underlying collection. The `aliases` property should return a mapping
    from each canonical name to zero or more valid aliases.

    With those methods defined, this class provides standard `__len__` and
    `__iter__` methods that operate on the canonical names, as well as a
    `__contains__` method that returns true if a given string matches a
    canonical name or a known alias. This class also provides additional
    properties and methods for working with aliases.
    """
    def __contains__(self, key: str) -> bool:
        """Whether or not the key is a valid name or alias.

        By definition, the `None` object is not a valid name or alias in any
        collection of references. This ensures that algorithms which
        programmatically check for inclusion based on the results of methods
        like `dict.get` won't end up with a false positive.
        """
        if key is None:
            return False
        return key in self.labels

    def __len__(self) -> int:
        """The length of this collections."""
        return len(self.names)

    def __iter__(self):
        """Iterate over members of this collection."""
        return iter(self.names)

    @property
    @abc.abstractmethod
    def names(self) -> List[str]:
        """The canonical names in this collection."""
        pass

    @property
    @abc.abstractmethod
    def aliases(self) -> Mapping[str, Iterable[str]]:
        """A mapping from canonical name to valid alias(es)."""
        pass

    @property
    def labels(self) -> List[str]:
        """All names and aliases in this collection.

        This property is equivalent to `self.names + self.aliases` and simply
        exists to serve as a convenient shorthand.
        """
        all_aliases = [
            v for s in self.names
            for v in self.aliases[s]
        ]
        return self.names + all_aliases

    def translate(self, key: str) -> Optional[str]:
        """Get the canonical name for this key, if possible."""
        if key in self.names:
            return key
        for name in self.names:
            if key in self.aliases[name]:
                return name


class AliasedCollection(CollectionMixin, collections.abc.Collection):
    """An abstract collection of items with possible aliases.
    
    Implementations of this class must define the `_get_aliases_for` method.
    """
    def __init__(self, collection: Collection) -> None:
        self._collection = collection
        self.collect('_collection')
        self._names = None
        self._aliases = None

    @abc.abstractmethod
    def _get_aliases_for(self, key: str) -> List[str]:
        """Get the alias(es) corresponding to the given key."""
        pass

    def __contains__(self, key: str) -> bool:
        """Whether or not the key is a valid name or alias.

        By definition, the `None` object is not a valid name or alias in any
        collection of references. This ensures that algorithms which
        programmatically check for inclusion based on the results of methods
        like `dict.get` won't end up with a false positive.
        """
        if key is None:
            return False
        return key in self.names or key in self.aliases

    @property
    def names(self) -> List[str]:
        """The valid names in this collection."""
        if self._names is None:
            self._names = list(self._collection)
        return self._names

    @property
    def aliases(self) -> List[str]:
        """All aliases in this collection, in a single list."""
        if self._aliases is None:
            self._aliases = [
                v for s in self._collection
                for v in self._get_aliases_for(s)
            ]
        return self._aliases

    @property
    def labels(self) -> List[str]:
        """All names and aliases in this collection.

        This property is equivalent to `self.names + self.aliases` and simply
        exists to serve as a convenient shorthand.
        """
        return self.names + self.aliases

    def translate(self, key: str) -> Optional[str]:
        """Get the canonical name for this key, if possible."""
        if key in self._collection:
            return key
        for name in self._collection:
            if key in self._get_aliases_for(name):
                return name


class AliasedKeyError(Exception):
    """Key error for aliased collections."""
    def __init__(self, key: str) -> None:
        self.key = key

    def __str__(self) -> str:
        return f"'{self.key}' is not a known name or alias."


Aliases = TypeVar('Aliases', bound=Union[str, Iterable[str]])
Aliases = Union[str, Iterable[str]]

class AliasedKey(CollectionMixin, collections.abc.Collection):
    """A type that supports aliased-mapping keys."""

    __slots__ = ('_aliases')

    _builtin = (tuple, list, set)

    def __init__(self, *a: Aliases) -> None:
        if not a:
            raise TypeError("At least one alias is required") from None
        self._aliases = set(
            a[0] if isinstance(a[0], (self._builtin, AliasedKey))
            else a
        )
        self.collect('_aliases')

    def __hash__(self) -> int:
        """Compute the hash of the underlying key tuple."""
        return hash(tuple(self._aliases))

    def __eq__(self, other: 'AliasedKey') -> bool:
        """True if two instances' key tuples are equal."""
        if isinstance(other, AliasedKey):
            return self._aliases == other._aliases
        if isinstance(other, self._builtin):
            return self._aliases == set(other)
        return False

    def __getitem__(self, index) -> str:
        """Get an alias by index."""
        # This implicitly passes the responsibility for handling negative
        # indices, raising exceptions, etc. down to the `tuple` class.
        return self._aliases[index]

    def __add__(self, other: Union[Aliases, 'AliasedKey']):
        """Combine these aliases with `other`."""
        if isinstance(other, str):
            other = (other,)
        return self._type(set(self) | set(other))

    def __sub__(self, other: Union[Aliases, 'AliasedKey']):
        """Remove `other` from these aliases."""
        if isinstance(other, str):
            other = (other,)
        return self._type(set(self) - set(other))

    @property
    def _type(self):
        """Internal representation of the current type."""
        return type(self)

    def __repr__(self) -> str:
        """An unambiguous representation of this instance."""
        return f"{self.__class__.__qualname__}({self._display!r})"

    def __str__(self) -> str:
        """A printable representation of this instance."""
        return str(self._display)

    @property
    def _display(self) -> str:
        """Internal helper for `__repr__` and `__str__`."""
        return ' | '.join(self._aliases)


_VT = TypeVar('_VT')


class AliasedMapping(CollectionMixin, collections.abc.Mapping, ReprStrMixin):
    """A mapping class that supports aliased keys.
    
    Parameters
    ----------
    mapping : mapping
        An object that maps strings or iterables of strings to values of any
        type.

    Examples
    --------
    Create an instance from a standard `dict` with strings or tuples of strings
    as keys.

    >>> mapping = {'a': 1, ('b', 'B'): 2, ('c', 'C', 'c0'): -9}
    >>> aliased = AliasedMapping(mapping)
    >>> aliased
    AliasedMapping('a': 1, 'b | B': 2, 'c | C | c0': -9)

    The caller may access an individual item by any one (but only one) of its
    valid aliases.

    >>> aliased['a']
    1
    >>> aliased['b']
    2
    >>> aliased['B']
    2
    >>> aliased[('b', 'B')]
    ...
    KeyError: ('b', 'B')

    Aliased items are identical.

    >>> aliased['b'] is aliased['B']
    True

    The representation of keys, values, and items reflect the internal grouping
    by alias, but iterating over each produces de-aliased members. This behavior
    naturally supports loops and comprehensions, since access is only valid for
    individual items.

    >>> aliased.keys()
    AliasedKeys(['a', 'b | B', 'c | C | c0'])
    >>> aliased.values()
    AliasedValues([1, 2, -9])
    >>> aliased.items()
    AliasedItems([('a', 1), ('b | B', 2), ('c | C | c0', -9)])
    >>> list(aliased.keys())
    ['a', 'b', 'B', 'c', 'C', 'c0']
    >>> list(aliased.values())
    [1, 2, 2, -9, -9, -9]
    >>> list(aliased.items())
    [('a', 1), ('b', 2), ('B', 2), ('c', -9), ('C', -9), ('c0', -9)]
    >>> for k, v in aliased.items():
    ...     print(k, aliased[k], v)
    ... 
    a 1 1
    b 2 2
    B 2 2
    c -9 -9
    C -9 -9
    c0 -9 -9

    It is always possible to access the equivalent de-aliased `dict`.

    >>> aliased.flat
    {'a': 1, 'b': 2, 'B': 2, 'c': -9, 'C': -9, 'c0': -9}

    Updates and deletions apply to all associated aliases.

    >>> aliased['c'] = 5.6
    >>> list(aliased.items())
    [('a', 1), ('b', 2), ('B', 2), ('c', 5.6), ('C', 5.6), ('c0', 5.6)]
    >>> del aliased['c']
    >>> list(aliased.items())
    [('a', 1), ('b', 2), ('B', 2)]

    Users may access all aliases for a given key, or register new ones, via the
    `alias` method. Attempting to register an alias will raise a `KeyError` if
    it is already an alias for a different key.

    >>> aliased.alias('b')
    AliasedKey('B')
    >>> aliased.alias('b', include=True)
    AliasedKey('b | B')
    >>> aliased.alias(b='my B')
    >>> aliased.alias('b')
    AliasedKey('B | my B')
    >>> aliased
    AliasedMapping('a': 1, 'b | my B | B': 2)
    >>> aliased.alias(a='b')
    ...
    KeyError: "'b' is already an alias for '(B, my B)'"

    Notes
    -----
    The length of this object, as well as its keys, values, and items, is equal
    to the number of valid aliases it contains. This is consistent with the
    many-to-one nature of the mapping despite the fact that it internally stores
    aliases and values in a one-to-one mapping.

    """

    Aliasable = TypeVar('Aliasable', bound=Mapping)
    Aliasable = Mapping[Aliases, _VT]

    def __init__(
        self,
        mapping: Union[Aliasable, 'AliasedMapping']=None
    ) -> None:
        self._aliased = self._build_aliased(mapping)
        self.collect('_flat_keys')

    T = TypeVar('T')

    def _build_aliased(
        self,
        mapping: Union[Aliasable, 'AliasedMapping'],
    ) -> Dict[AliasedKey, _VT]:
        """Build a `dict` that maps aliased keys to user values."""
        if isinstance(mapping, AliasedMapping):
            return {
                key: value for key, value in mapping.items().aliased
            }
        _mapping = mapping or {}
        return {
            AliasedKey(key): value for key, value in _mapping.items()
        }

    @property
    def _flat_keys(self):
        """All keys in the mapping, as a single list."""
        return [key for keys in self._aliased.keys() for key in keys]

    @property
    def flat(self) -> Dict[str, _VT]:
        """Expand aliased items into a standard dictionary."""
        return {key: self[key] for key in self._flat_keys}

    @classmethod
    def of(
        cls,
        user: Mapping[str, Mapping[str, Any]],
        alias_key: str='aliases',
        value_key: str=None,
    ): # How do I annotate this so it's correct for subclasses?
        """Create an aliased mapping from another mapping, if possible.

        Note that this method operates differently from the standard class
        constructor. Whereas the constructor creates aliased keys from
        pre-grouped keys in the input mapping, this method expects keys in the
        input mapping to be single strings and creates aliased keys from the
        mappings to which those keys point (so-called "interior mappings").
        
        Parameters
        ----------
        user : mapping
            An object that maps string keys to interior mappings of strings to
            any type.

        alias_key : str, default='aliases'
            The key in the interior mappings whose values to use as aliases for
            the corresponding keys in the user-provided mapping. Absence of
            `alias_key` from a given interior mapping simply means the
            corresponding key in the user-provided mapping will be the only
            alias for that entry.

        value_key : str, default=None
            The key in the interior mappings whose values to extract as the sole
            aliased value in the result. If `value_key` is not present in a
            given interior mapping, this method will insert a default value of
            `None`. If `value_key` is `None`, this method will extract a
            dictionary of all key-value pairs not including `alias_key`.

        Returns
        -------
        aliased mapping
            A new instance of this class, with aliased keys and values taken
            from the user-provided mapping.

        See Also
        --------
        fromkeys : similar to `dict.fromkeys`

        Examples
        --------
        Create aliased mappings from a user dictionary with the default alias
        key and various value keys.

        >>> user = {
        ...     'a': {'aliases': ('A', 'a0'), 'n': 1, 'm': 'foo'},
        ...     'b': {'aliases': 'B', 'n': -4},
        ... }
        >>> aliased = AliasedMapping.of(user)
        >>> aliased
        AliasedMapping('a0 | A | a': {'n': 1, 'm': 'foo'}, 'b | B': {'n': -4})
        >>> aliased['a']
        {'n': 1, 'm': 'foo'}
        >>> aliased = AliasedMapping.of(user, value_key='n')
        >>> aliased
        AliasedMapping('a0 | A | a': 1, 'b | B': -4)
        >>> aliased['a']
        1
        >>> aliased = AliasedMapping.of(user, value_key='m')
        >>> aliased
        AliasedMapping('a0 | A | a': foo, 'b | B': None)
        >>> aliased['a']
        'foo'

        Create aliased mappings from a user dictionary, swapping roles of alias
        key and value key.
        >>> user = {
        ...     'a': {'foo': 'A', 'bar': 'a0'},
        ...     'b': {'foo': 'B', 'bar': 'b0'},
        ... }
        >>> aliased = AliasedMapping.of(user, alias_key='foo', value_key='bar')
        >>> aliased
        AliasedMapping('A | a': a0, 'b | B': b0)
        >>> aliased['a']
        'a0'
        >>> aliased = AliasedMapping.of(user, alias_key='bar', value_key='foo')
        >>> aliased
        AliasedMapping('a0 | a': A, 'b | b0': B)
        >>> aliased['a']
        'A'
        """
        if isinstance(user, AliasedMapping):
            return cls(user)
        keys = cls.extract_keys(user, alias_key=alias_key)
        if isinstance(value_key, str):
            values = [group.get(value_key) for group in user.values()]
        else:
            values = [
                {k: v for k, v in group.items() if k != alias_key}
                for group in user.values()
            ]
        d = {k: v for k, v in zip(keys, values)}
        return cls(d)

    @classmethod
    def fromkeys(
        cls,
        user: Mapping[str, Mapping[str, Any]],
        alias_key: str='aliases',
        value: Any=None,
    ): # How do I annotate this so it's correct for subclasses?
        """Create an aliased mapping based on another mapping's keys.
        
        This class method is essentially a special case of `AliasedMapping.of`.

        Parameters
        ----------
        user : mapping
            See `AliasedMapping.of`
        alias_key : string
            See `AliasedMapping.of`
        value : any
            The fill value to use for all items.

        Returns
        -------
        aliased mapping
            A new instance of this class, with aliased keys taken from the
            user-provided mapping and each value set to the given value.

        Examples
        --------
        Create an aliased mapping with keys taken from a pre-defined dictionary
        and all values set to -1.0

        >>> user = {
        ...     'a': {'aliases': ('A', 'a0'), 'n': 1, 'm': 'foo'},
        ...     'b': {'aliases': 'B', 'n': -4},
        ... }
        >>> aliased = AliasedMapping.fromkeys(user, value=-1.0)
        >>> aliased
        AliasedMapping('a0 | a | A': -1.0, 'B | b': -1.0)
        """
        keys = cls.extract_keys(user, alias_key=alias_key)
        d = {k: value for k in keys}
        return cls(d)

    @classmethod
    def extract_keys(
        cls,
        user: Mapping[str, Mapping[str, Any]],
        alias_key: str='aliases',
    ) -> List[AliasedKey]:
        """Extract keys for use in an aliased mapping.
        
        Parameters
        ----------
        user : mapping
            See `AliasedMapping.of`

        alias_key : string
            See `AliasedMapping.of`

        Examples
        --------
        >>> user = {
        ...     'a': {'aliases': ('A', 'a0'), 'n': 1, 'm': 'foo'},
        ...     'b': {'aliases': 'B', 'n': -4},
        ... }
        >>> keys = AliasedMapping.extract_keys(user)
        >>> keys
        [AliasedKey('a | A | a0'), AliasedKey('B | b')]
        """
        if isinstance(user, AliasedMapping):
            return user.keys().aliased
        return [
            AliasedKey(k) + AliasedKey(v.get(alias_key, ()))
            for k, v in user.items()
        ]

    def __getitem__(self, key: Union[str, AliasedKey]) -> _VT:
        """Look up a value by one of its keys."""
        if isinstance(key, AliasedKey):
            key = list(key)[0]
        if key in self._flat_keys:
            return self._aliased[self._resolve(key)]
        raise KeyError(key) from None

    def _resolve(self, key: str) -> AliasedKey:
        """Resolve `key` into an existing or new aliased key."""
        for aliased in self._aliased.keys():
            if key in aliased:
                return aliased
        return AliasedKey(key)

    def alias(self, *current, include=False):
        """Get the alias for an existing key or register new ones."""
        if current:
            if len(current) > 1:
                raise ValueError("Can only get one aliased key at a time")
            key = current[0]
            if include:
                return self._resolve(key)
            return self._resolve(key) - key

    def _not_available(self, key: str) -> NoReturn:
        """True if this key is not currently in use."""
        aliases = self.alias(key)
        this = ", ".join(f'{a}' for a in aliases)
        if len(aliases) > 1:
            this = f"({this})"
        raise KeyError(f"'{key}' is already an alias for {this!r}")

    def __eq__(self, other: Mapping) -> bool:
        """Define equality between this and another object."""
        if not isinstance(other, Mapping):
            return NotImplemented
        if isinstance(other, AliasedMapping):
            return self.items() == other.items()
        return dict(self.items()) == dict(other.items())

    def __or__(self, other: 'AliasedMapping'):
        """Merge this aliased mapping with another."""
        items = (*self.items().aliased, *other.items().aliased)
        tmp = {k: v for k, v in items}
        return type(self)(tmp)

    def __str__(self) -> str:
        """A simplified representation of this instance."""
        return ', '.join(
            f"'{g}': {v!r}"
            for g, v in zip(self._aliased.keys(), self._aliased.values())
        )

    def keys(self):
        """A view on this instance's keys."""
        return AliasedKeysView(self)

    def values(self):
        """A view on this instance's values."""
        return AliasedValuesView(self)

    def items(self):
        """A view on this instance's key-value pairs."""
        return AliasedItemsView(self)

    def copy(self):
        """Create a shallow copy of this instance."""
        return type(self)(self._aliased)

    __class_getitem__ = classmethod(type(List[int]))


class AliasedKeysView(collections.abc.KeysView):
    """A view on the keys of an aliased mapping.
    
    Notes
    -----
    See note on lengths at `~AliasedMapping`.
    """

    def __init__(self, mapping: AliasedMapping) -> None:
        super().__init__(mapping)
        self._aliased = mapping._aliased

    def __eq__(self, other: KeysView) -> bool:
        """The definition of equality between this and other keys views."""
        def equal(this, that) -> bool:
            """Helper function to checking equality."""
            return (
                len(that) == len(this)
                and
                all(key in this for key in that)
            )
        if not isinstance(other, KeysView):
            return NotImplemented
        this = tuple(self.aliased)
        if isinstance(other, AliasedKeysView):
            return equal(this, tuple(other.aliased))
        return equal(this, tuple(other))

    @property
    def aliased(self):
        """An iterator over groups of aliased keys."""
        return iter(self._aliased)

    def __repr__(self) -> str:
        """An unambiguous representation of this instance."""
        keys = [str(keys) for keys in self._aliased]
        return f"AliasedKeys({keys})"


class AliasedValuesView(collections.abc.ValuesView):
    """A view on the values of an aliased mapping.

    Notes
    -----
    See note on lengths at `~AliasedMapping`.
    """

    def __init__(self, mapping: AliasedMapping) -> None:
        super().__init__(mapping)
        self._aliased = mapping._aliased

    def __eq__(self, other: ValuesView) -> bool:
        """The definition of equality between this and other values views."""
        def equal(this, that) -> bool:
            """Helper function to checking equality."""
            return (
                len(that) == len(this)
                and
                all(value in this for value in that)
            )
        if not isinstance(other, ValuesView):
            return NotImplemented
        this = tuple(self.aliased)
        if isinstance(other, AliasedValuesView):
            return equal(this, tuple(other.aliased))
        return equal(this, tuple(other))

    @property
    def aliased(self):
        """An iterator over groups of aliased values."""
        return iter(self._aliased.values())

    def __repr__(self) -> str:
        """An unambiguous representation of this instance."""
        return f"AliasedValues({list(self._aliased.values())})"


class AliasedItemsView(collections.abc.ItemsView):
    """A view on the key-value pairs of an aliased mapping.
    
    Notes
    -----
    See note on lengths at `~AliasedMapping`.
    """

    def __init__(self, mapping: AliasedMapping) -> None:
        super().__init__(mapping)
        self._aliased = mapping._aliased

    def __eq__(self, other: ItemsView) -> bool:
        """The definition of equality between this and other items views."""
        def equal(this, that) -> bool:
            """Helper function to checking equality."""
            return (
                len(that) == len(this)
                and
                all(key in this for key in that)
            )
        if not isinstance(other, ItemsView):
            return NotImplemented
        this = tuple(self.aliased)
        if isinstance(other, AliasedItemsView):
            return equal(this, tuple(other.aliased))
        return equal(this, tuple(other))

    @property
    def aliased(self):
        """An iterator over groups of aliased items."""
        return iter(self._aliased.items())

    def __repr__(self) -> str:
        """An unambiguous representation of this instance."""
        pairs = [(str(keys), value) for keys, value in self._aliased.items()]
        return f"AliasedItems({pairs})"


class AliasedMutableMapping(AliasedMapping, collections.abc.MutableMapping):
    """A mutable version of `AliasedMapping`.
    
    Parameters
    ----------
    mapping : mapping
        An object that maps strings or iterables of strings to values of any
        type.

    Examples
    --------
    These examples build on the examples shown in `AliasedMapping`.

    >>> mutable = iterables.AliasedMutableMapping(aliased)

    Updates and deletions apply to all associated aliases.

    >>> mutable['c'] = 5.6
    >>> list(mutable.items())
    [('a', 1), ('b', 2), ('B', 2), ('c', 5.6), ('C', 5.6), ('c0', 5.6)]
    >>> del mutable['c']
    >>> list(mutable.items())
    [('a', 1), ('b', 2), ('B', 2)]

    Users may access all aliases for a given key, or register new ones, via the
    `alias` method. Attempting to register an alias will raise a `KeyError` if
    it is already an alias for a different key.

    >>> mutable.alias('b')
    AliasedKey('B')
    >>> mutable.alias('b', include=True)
    AliasedKey('b | B')
    >>> mutable.alias(b='my B')
    >>> mutable.alias('b')
    AliasedKey('B | my B')
    >>> mutable
    AliasedMapping('a': 1, 'b | my B | B': 2)
    >>> mutable.alias(a='b')
    ...
    KeyError: "'b' is already an alias for '(B, my B)'"

    Notes
    -----
    See note on lengths at `~AliasedMapping`.
    """

    def __setitem__(self, key: str, value: _VT):
        """Assign a value to `key` and its aliases."""
        self._aliased[self._resolve(key)] = value

    def __delitem__(self, key: str):
        """Remove the item corresponding to `key`."""
        try:
            del self._aliased[self._resolve(key)]
        except KeyError:
            raise KeyError(key) from None

    def alias(self, *current, include=False, **new: str):
        """Get the alias for an existing key or register new ones."""
        if current and new:
            raise TypeError("Can't get and set aliases at the same time")
        if current:
            if len(current) > 1:
                raise ValueError("Can only get one aliased key at a time")
            key = current[0]
            if include:
                return self._resolve(key)
            return self._resolve(key) - key
        for key, alias in new.items():
            if alias:
                if alias in self._flat_keys:
                    self._not_available(alias)
                updated = self._resolve(key) + alias
                self._aliased[updated] = self[key]
                del self[key]


class NameMap(MappingBase):
    """A mapping from aliases to canonical names.
    
    Notes
    -----
    See note on lengths at `~AliasedMapping`.
    """

    def __new__(cls, refs, *args, **kwargs):
        try:
            iter(refs)
        except TypeError:
            raise TypeError("Reference names must be iterable") from None
        return super().__new__(cls)

    _DefinesAliases = TypeVar(
        '_DefinesAliases',
        Iterable[Iterable[str]],
        Mapping[str, Iterable[str]],
        Mapping[str, Mapping[str, Iterable[str]]]
    )
    _DefinesAliases = Union[
        Iterable[Iterable[str]],
        Mapping[str, Iterable[str]],
        Mapping[str, Mapping[str, Iterable[str]]],
    ]

    def __init__(
        self,
        refs: Union[Iterable[str], Mapping[str, Any]],
        defs: _DefinesAliases,
        key: str='aliases',
    ) -> None:
        names = refs.keys() if isinstance(refs, Mapping) else refs
        self._mapping = self._build_mapping(names, defs, key)
        super().__init__(self._mapping.keys())
        self._init = {'refs': refs, 'defs': defs, 'key': key}

    def _build_mapping(self, names, aliases, key):
        """Build the internal mapping from aliases to canonical names.

        This method first creates an identity map of canonical
        names (i.e., `name` -> `name`), with which it spawns a trivial aliased
        mapping. It then determines the appropriate aliases for each canonical
        name and updates the aliased-mapping keys. The result is a mapping from
        one or more aliases to a canonical name. In case there are no aliases
        associated with a canonical name, its aliased key will simply contain
        itself.
        """
        identity = {name: name for name in names}
        namemap = AliasedMutableMapping(identity)
        updates = self._get_aliases(names, aliases, key)
        namemap.alias(**updates)
        return AliasedMapping(namemap)

    def _get_aliases(self, names, defs: _DefinesAliases, key):
        """Determine the appropriate aliases for each canonical name."""
        # Mapping <: Iterable, so we need to check Mapping first.
        if isinstance(defs, Mapping):
            # There are two allowed types of Mapping:
            # 1) Mapping[str, Mapping[str, Iterable[str]]]
            # 2) Mapping[str, Iterable[str]]
            
            # Make sure the keys are all strings.
            if any(k for k in defs if not isinstance(k, str)):
                raise TypeError("All aliases must be strings") from None
            # Again, we need to check Mapping values before Iterable values.
            if all(isinstance(v, Mapping) for v in defs.values()):
                return {
                    name: tuple(v.get(key, ()))
                    for name, v in defs.items()
                }
            if all(isinstance(v, Iterable) for v in defs.values()):
                return {name: tuple(aliases) for name, aliases in defs.items()}
        # Alias definitions are in a non-mapping iterable. We may want to
        # further check that each member of `defs` is itself an iterable of
        # strings.
        only_iterables = all(isinstance(d, Iterable) for d in defs)
        if isinstance(defs, Iterable) and only_iterables:
            return {
                name: tuple(aliases)
                for aliases in defs
                for name in names
                if name in aliases
            }
        return {}

    def invert(self):
        """Invert the mapping from {aliases -> name} to {name -> aliases}."""
        inverted = {
            name: list(aliases)
            for aliases, name in self._mapping.items().aliased
        }
        self._mapping = AliasedMapping(inverted)
        return self

    def __getitem__(self, key: str):
        """Get the canonical name for `key`."""
        if key in self._mapping:
            return self._mapping[key]
        raise AliasedKeyError(key)

    def __str__(self) -> str:
        """A simplified representation of this object."""
        items = {f"{k}: '{v}'" for k, v in self._mapping.items().aliased}
        return ', '.join(items)

    def __repr__(self) -> str:
        """An unambiguous representation of this object."""
        return f"{self.__class__.__qualname__}({self})"

    def keys(self):
        """A view on this object's aliased keys."""
        return AliasedKeysView(self._mapping)

    def values(self):
        """A view on this object's aliased values."""
        return AliasedValuesView(self._mapping)

    def items(self):
        """A view on this object's aliased key-value pairs."""
        return AliasedItemsView(self._mapping)

    def copy(self):
        """Make a shallow copy of this object."""
        return type(self)(**self._init)


class AliasedCache(CollectionMixin, collections.abc.Mapping):
    """An aliased mapping that remembers requested items."""

    Value = TypeVar('Value')

    def __init__(
        self,
        mapping: Mapping[str, Value],
        aliases: Union[Iterable[Iterable[str]], Mapping[str, Iterable[str]]],
        fill: Any=None,
    ) -> None:
        self._datamap = mapping
        self._fill = fill
        self.names = tuple(self._datamap.keys())
        self.collect('names')
        namemap = self._build_namemap(aliases)
        self._namemap = namemap
        self._cache = AliasedMutableMapping.fromkeys(namemap)

    def _build_namemap(self, aliases):
        """Internal mapping from aliases to canonical names."""
        trivial = {name: name for name in self.names}
        namemap = AliasedMutableMapping(trivial)
        updates = self._update_keys(aliases)
        namemap.alias(**updates)
        return AliasedMapping(namemap)

    def _update_keys(self, aliases):
        """Get appropriate aliased-key updates based on type of `aliases`."""
        if not isinstance(aliases, (Mapping, Iterable)):
            return {}
        if isinstance(aliases, Mapping):
            return {name: tuple(alias) for name, alias in aliases.items()}
        # This check would be more thorough if we had could use something like
        # AliasedCollection to guarantee that `aliases` is an iterable of
        # iterables of strings.
        if isinstance(aliases, Iterable):
            return {
                key: alias
                for alias in aliases
                for key in alias
                if key in self.names and not isinstance(alias, str)
            }

    def __getitem__(self, key: str) -> Value:
        """Aliased access to the underlying data."""
        if key in self._namemap:
            current = self._cache.get(key, self._fill)
            if current != self._fill:
                return current
            name = self._namemap[key]
            data = self._datamap[name]
            self._cache[key] = data
            return data
        raise AliasedKeyError(key) from None

    def keys(self):
        return self._cache.keys()

    def values(self):
        return self._cache.values()

    def items(self):
        return self._cache.items()

    def __str__(self) -> str:
        items = (f"'{k}': {v!r}" for k, v in self._cache.items().aliased)
        return ', '.join(items)

    def __repr__(self) -> str:
        return f"{self.__class__.__qualname__}({self})"


class ObjectRegistry(CollectionMixin, collections.abc.Mapping):
    """A class for associating metadata with abitrary objects."""
    def __init__(self, base: Mapping=None, object_key: str='object') -> None:
        mapping = base or {}
        self._items = {
            k: {object_key: v} if not isinstance(v, Mapping) else v
            for k, v in mapping.items()
        }
        self._object_key = object_key
        self._init = self._items.copy()
        self._default_key_count = 1
        self.collect('_items')

    def register(
        self,
        _obj=None,
        name: str=None,
        overwrite: bool=False,
        **metadata
    ) -> Any:
        """Register an object and any associated metadata.

        This function exists to decorate objects. Without any arguments, it will
        log the decorated object in an internal mapping, keyed by the object's
        name. The user may optionally provide key-value pairs of metadata to
        associate with the object.

        Positional Parameters
        ---------------------
        None

        Keyword Parameters
        ------------------
        name : string

        The name to use as this object's key in the internal mapping. The
        default is `None`, which causes this class to create a unique key based
        on the defined name of the object.

        overwrite: bool

        If true and there is already an object with the key given by `name`,
        overwrite that object. Default is false. This keyword has no effect if
        `name` is `None`.

        **metadata : key-value pairs

        Arbitrary metadata to associate with the decorated object.

        Returns
        -------
        Any

        The decorated object.

        Examples
        --------
        Create an empty object registry, register functions with and without
        metadata, and reset the registry.

        >>> from goats.common.iterables import ObjectRegistry
        >>> from pprint import pprint
        >>> registry = ObjectRegistry()
        >>> @registry.register
        ... def myfunc():
        ...     pass
        ... 
        >>> pprint(registry.mapping)
        {'myfunc': {'object': <function myfunc at 0x7f6097427ee0>}}
        >>> @registry.register(units='cm/s')
        ... def vel():
        ...     pass
        ... 
        >>> pprint(registry.mapping)
        {'myfunc': {'object': <function myfunc at 0x7f6097427ee0>},
        'vel': {'object': <function vel at 0x7f60570fb280>, 'units': 'cm/s'}}
        >>> registry.reset()
        >>> pprint(registry.mapping)
        {}

        Create a registry with an existing `dict` and register a function with
        the same name as an object in the initializing `dict`.

        >>> from goats.common.iterables import ObjectRegistry
        >>> from pprint import pprint
        >>> mymap = {'this': [2, 3]}
        >>> registry = ObjectRegistry(mymap)
        >>> @registry.register
        ... def this():
        ...     pass
        ... 
        >>> pprint(registry.mapping)
        {'this': {'object': [2, 3]},
        'this_1': {'object': <function this at 0x7f46d49961f0>}}

        Create a registry with an existing `dict` and register a function with a
        defined name that is the same as an object in the initializing `dict`,
        but store it under a different name, then repeat the process but
        overwrite the first decorated function.

        >>> from goats.common.iterables import ObjectRegistry
        >>> from pprint import pprint
        >>> mymap = {'this': [2, 3]}
        >>> registry = ObjectRegistry(mymap)
        >>> @registry.register(name='that')
        ... def this():
        ...     pass
        >>> pprint(registry.mapping)
        {'that': {'object': <function this at 0x7f1fab723d30>},
        'this': {'object': [2, 3]}}
        >>> @registry.register(name='that', overwrite=True)
        ... def other():
        ...     pass
        >>> pprint(registry.mapping)
        {'that': {'object': <function other at 0x7f1fab723dc0>},
        'this': {'object': [2, 3]}}
        """
        @functools.wraps(_obj)
        def decorator(obj):
            key = self._get_mapping_key(obj, user=name, overwrite=overwrite)
            self._items[key] = {self._object_key: obj, **metadata}
            return obj
        if _obj is None:
            return decorator
        return decorator(_obj)

    def _get_mapping_key(
        self,
        obj: Any,
        user: str=None,
        overwrite: bool=False
    ) -> str:
        """Get an appropriate key to associate with this object."""
        available = user not in self._items or overwrite
        if user and isinstance(user, Hashable) and available:
            return user
        return self._get_default_key(obj)

    def _get_default_key(self, obj: Any) -> str:
        """Get a default key based on the object's name and existing keys."""
        options = ['__qualname__', '__name__']
        for option in options:
            if hasattr(obj, option):
                proposed = getattr(obj, option)
                if proposed not in self._items:
                    return proposed
                new_default_key = f"{proposed}_{self._default_key_count}"
                self._default_key_count += 1
                return new_default_key

    def reset(self) -> None:
        """Reset the internal mapping to its initial state."""
        self._items = {**self._init}

    def __getitem__(self, key: str) -> Dict[str, Any]:
        """Get an item from the object collection."""
        return self._items[key]

    def keys(self) -> AbstractSet[str]:
        """Mapping keys. Defined here just to specify return type."""
        return super().keys()

    def values(self) -> ValuesView[Dict[str, Any]]:
        """Mapping values. Defined here just to specify return type."""
        return super().values()

    def items(self) -> AbstractSet[Tuple[str, Dict[str, Any]]]:
        """Mapping items. Defined here just to specify return type."""
        return super().items()

    def copy(self) -> 'ObjectRegistry':
        """A shallow copy of this instance."""
        return ObjectRegistry(self._items.copy(), object_key=self._object_key)

    def __repr__(self) -> str:
        """An unambiguous representation of this object."""
        return f"{self.__class__.__qualname__}({self._items})"


class BinaryGroups:
    """A class that creates groups of included and excluded items."""

    _KT = TypeVar('_KT')
    _VT = TypeVar('_VT')

    def __init__(
        self,
        items: Union[Iterable[_KT], Mapping[_KT, _VT]],
        sorter: Container[_KT],
    ) -> None:
        self._items = items
        self._sorter = sorter
        self._included = None
        self._excluded = None
        self._sort_by(sorter)

    def sorter(self, new: Container=None) -> Optional['BinaryGroups']:
        """Get or set the definition of inclusion."""
        if new is None:
            return self._sorter
        self._sort_by(new)
        return self

    def _sort_by(self, container: Container) -> None:
        """Sort the items into groups."""
        if isinstance(self._items, Mapping):
            self._included = {
                k: v for k, v in self._items.items() if k in container
            }
            self._excluded = {
                k: v for k, v in self._items.items() if k not in container
            }
        else:
            self._included = [
                item for item in self._items if item in container
            ]
            self._excluded = [
                item for item in self._items if item not in container
            ]

    @property
    def included(self) -> Union[Tuple[_KT], Dict[_KT, _VT]]:
        """The items included in the given container."""
        if isinstance(self._items, Mapping):
            return self._included
        return tuple(self._included)

    @property
    def excluded(self) -> Union[Tuple[_KT], Dict[_KT, _VT]]:
        """The items excluded from the given container."""
        if isinstance(self._items, Mapping):
            return self._excluded
        return tuple(self._excluded)


class TableKeyError(KeyError):
    """No common key with this name."""
    def __str__(self) -> str:
        if len(self.args) > 0:
            return f"Table has no common key '{self.args[0]}'"
        return "Key not found in table"


class TableValueError(Exception):
    """An exception occurred during value-based look-up."""

    def __init__(self, value: Any) -> None:
        self.value = value


class AmbiguousValueError(TableValueError):
    """Failed to find a unique entry by value."""

    def __str__(self) -> str:
        return f"No unique entry containing {self.value}"


class MissingValueError(TableValueError):
    """Failed to find any qualifying entries by value."""

    def __str__(self) -> str:
        return f"No entries containing {self.value}"


class TableRequestError(Exception):
    """An exception occurred during standard look-up."""

    def __init__(self, request: Mapping) -> None:
        self.request = request


class AmbiguousRequestError(TableRequestError):
    """There are multiple instances of the same value for this key."""

    def __str__(self) -> str:
        """Print the requested pairs in the order provided."""
        items = list(self.request.items())
        n_items = len(items)
        if n_items <= 2:
            requested = " and ".join(f"'{k}={v}'" for k, v in items)
        else:
            these = ", ".join(f"'{k}={v}'" for k, v in items[:-1])
            this = f"'{items[-1][0]}={items[-1][1]}'"
            requested = f"{these}, and {this}"
        if n_items == 1:
            return f"The search criterion {requested} is ambiguous"
        return f"The search criteria {requested} are ambiguous"


class TableLookupError(TableRequestError):
    """Could not find the requested key-value pair(s)."""

    def __str__(self) -> str:
        """Print the requested pairs in the order provided."""
        items = list(self.request.items())
        if len(items) <= 2:
            criteria = " and ".join(f"{k}={v}" for k, v in items)
        else:
            these = ", ".join(f"{k}={v}" for k, v in items[:-1])
            this = f"{items[-1][0]}={items[-1][1]}"
            criteria = f"{these}, and {this}"
        return f"Table has no entry with {criteria}"


class Table(MappingBase):
    """A collection of mappings with support for multi-key search.

    Subclasses of this class may override the `_parse` method to customize
    parsing user input into a key-value pair, and the `_prepare` method to
    modify the search result (e.g., cast it to a custom type) before
    returning.
    """

    _KT = TypeVar('_KT', bound=str)
    _VT = TypeVar('_VT')
    _ET = TypeVar('_ET', bound=Mapping)

    def __init__(self, entries: Collection[_ET]) -> None:
        super().__init__(entries)
        self._entries = entries
        self._keys = None

    def show(self, names: Iterable[str]=None):
        """Print a formatted version of this table."""
        colpad = 2
        names = names or ()
        columns = {
            name: {'width': len(name)}
            for name in names if name in self.keys
        }
        dashed = {
            name: '-' * column['width']
            for name, column in columns.items()
        }
        for name, column in columns.items():
            values = self.get(name, ())
            for value in values:
                column['width'] = max(column['width'], len(str(value)))
        headers = [
            f"{name:^{2*colpad + column['width']}}"
            for name, column in columns.items()
        ]
        lines = [
            f"{dashed[name]:^{2*colpad + column['width']}}"
            for name, column in columns.items()
        ]
        rows = [
            [
                f"{entry[name]:^{2*colpad + column['width']}}"
                for name, column in columns.items()
            ] for entry in self
        ]
        if headers:
            print(''.join(header for header in headers if header))
        if lines:
            print(''.join(line for line in lines if line))
        for row in rows:
            if row:
                print(''.join(row))

    @property
    def keys(self) -> Set[_KT]:
        """All the keys common to the individual mappings."""
        if self._keys is None:
            all_keys = [list(entry.keys()) for entry in self._entries]
            self._keys = set(all_keys[0]).intersection(*all_keys[1:])
        return self._keys

    def __getitem__(self, key: _KT) -> Tuple[_VT]:
        """Get all the values for a given key if it is common."""
        if key in self.keys:
            values = [entry[key] for entry in self._entries]
            return tuple(values)
        raise TableKeyError(key)

    def get(self, key: _KT, default: Any=None) -> Tuple[_VT]:
        """Get all the values for a given key when available."""
        try:
            return self[key]
        except TableKeyError:
            values = [entry.get(key, default) for entry in self._entries]
            return tuple(values)

    def find(
        self,
        value: _VT,
        unique: bool=False,
    ) -> Union[_ET, List[_ET]]:
        """Find entries with the given value."""
        found = [entry for entry in self._entries if value in entry.values()]
        if not found:
            raise MissingValueError(value)
        if not unique:
            return found
        if len(found) > 1:
            raise AmbiguousValueError(value)
        return found[0]

    def __call__(self, strict: bool=False, **request):
        """Look up an entry by user-requested keys.

        This method will try to return the unique entry in the instance
        collection that contains the given key(s) with the corresponding
        value(s). The iterative search (`strict=False`; default) will iterate
        through the key-value pairs until it either finds a unique entry or runs
        out of pairs. The strict search (`strict=True`) will attempt to find the
        unique entry with all the key-value pairs. Both searches will raise a
        `TableLookupError` if they fail to find a unique entry, or a
        `TableKeyError` if one of the given keys does not exist in all table
        entries. The iterative search will raise an `AmbiguousRequestError` if the
        given key-value pairs are insufficient to determine a unique entry.

        Parameters
        ----------
        strict : bool, default=False
            Fail if any of the given key-value pairs is not in the collection.

        **request : mapping
            Key-value pairs that define the search criteria. Each key must appear in all table entries.

            Key-value pairs in which each requested key is a key in every entry
            of the collection. Each requested value may correspond to its
            respective key in zero or more entries for an iterative search, but
            must correspond to its respective key in at least one entry for a
            strict search.

        Returns
        -------
        mapping
            The unique entry containing at least one of the requested key-value
            pairs (iterative search) or all of the requested key-value pairs
            (strict search).

        Raises
        ------
        TableKeyError
            A requested key is not present in every entry.
        
        TableLookupError
            Could not find an entry that matched the requested key-value pairs.

        AmbiguousRequestError
            The given key-value pair is ambiguous. Only applies to iterative
            search.
        """
        subset = [*self._entries]
        for n_checked, pair in enumerate(request.items(), start=1):
            key, value = pair
            if key not in self.keys:
                raise TableKeyError(key)
            subset = [
                entry for entry in subset
                if entry[key] == value
            ]
            if not strict:
                count = self[key].count(value)
                if count > n_checked and len(request) == n_checked:
                    raise AmbiguousRequestError(request)
                if len(subset) == 1:
                    return subset[0]
        if strict and len(subset) == 1:
            return subset[0]
        raise TableLookupError(request)


class Singleton:
    """A simple base class for creating singletons."""

    _exists = False
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._exists:
            return cls._instance
        new = super(Singleton, cls).__new__(cls)
        cls._exists = True
        cls._instance = new
        return new


class NothingType(Singleton):
    """An object that represents nothing in a variety of ways."""

    def __getitem__(self, index: Any) -> None:
        """Return `None`, regardless of `index`."""
        return None

    def __call__(self, *args, **kwargs) -> None:
        """Return `None`, regardless of input."""
        return None

    def __len__(self) -> int:
        """This object has zero length."""
        return 0

    def __contains__(self, key: str) -> bool:
        """This object contains nothing."""
        return False

    def __iter__(self) -> Iterable:
        """Return an empty iterator.
        
        Thanks to https://stackoverflow.com/a/36658865/4739101 for this one.
        """
        yield from ()

    def __next__(self):
        """There is always nothing left."""
        raise StopIteration

    def __bool__(self) -> bool:
        """This object is always false."""
        return False

Nothing = NothingType()
"""A unique object that represents nothing."""


class NonStrIterable(abc.ABCMeta):
    """A type representing a non-string iterable."""

    def __instancecheck__(cls, this: Any) -> bool:
        """True if `this` is not string-like and is iterable."""
        return not isinstance(this, (str, bytes)) and isinstance(this, Iterable)


class SeparableTypeError(TypeError):
    """Non-separable argument type."""

    def __init__(self, arg) -> None:
        self.arg = arg

    def __str__(self) -> str:
        return f"{self.arg!r} is not separable"


class Separable(collections.abc.Collection, metaclass=NonStrIterable):
    """A collection of independent members.

    This class represents iterable collections with members that have meaning
    independent of any other members. For example, a list of numbers is
    separable whereas a string is not, despite the fact that both objects are
    iterable collections.

    The motivation for this distinction is to make it easier to treat single
    numbers and strings equivalently to iterables of numbers and strings.
    """

    def __init__(self, arg) -> None:
        """Initialize a separable object from `arg`"""
        self.arg = self.parse(arg)

    @staticmethod
    def parse(arg):
        """Convert `arg` into a separable object.

        For most cases, this method will try to iterate over `arg`. If that
        operation succeeds, it will simply return `arg`; if the attempt to
        iterate raises a `TypeError`, it will assume that `arg` is a scalar and
        will return a one-element list containing `arg`. If `arg` is `None`,
        this method will return an empty list. If `arg` is a string, this
        method will return a one-element list containing `arg`.
        """
        if arg is None:
            return []
        if isinstance(arg, str):
            return [arg]
        try:
            iter(arg)
        except TypeError:
            return [arg]
        else:
            return arg

    def __iter__(self) -> Iterator[NonStrIterable]:
        return iter(self.arg)

    def __len__(self) -> int:
        return len(self.arg)

    def __contains__(self, this: object) -> bool:
        return this in self.arg

    def __eq__(self, other: 'Separable') -> bool:
        """True if two separable iterables have equal arguments."""
        if isinstance(other, Separable):
            return sorted(self) == sorted(other)
        return NotImplemented

    def __str__(self) -> str:
        """A simplified representation of this object."""
        return str(self.arg)

    def __repr__(self) -> str:
        """An unambiguous representation of this object."""
        return f"{self.__class__.__qualname__}({self})"

    __class_getitem__ = classmethod(type(List[int]))


def distribute(a, b):
    """Distribute `a` and `b` over each other.

    If both `a` and `b` are separable (see the `Separable` class), this function
    will return their Cartesian product. If only `a` or `b` is separable, this
    function will pair the non-separable argument with each element of the
    separable argument. If neither is separable, this function will raise an
    error.
    """
    a_separable = isinstance(a, Separable)
    b_separable = isinstance(b, Separable)
    if not (a_separable or b_separable):
        raise TypeError("At least one argument must be separable")
    if a_separable and b_separable:
        return iter(product(a, b))
    if not a_separable:
        return iter((a, i) for i in b)
    return iter((i, b) for i in a)


class classproperty(property):
    """A descriptor decorator to create a read-only class property.

    Adapted from https://stackoverflow.com/a/13624858/4739101.

    Note that `@classmethod` can wrap other descriptors in Python 3.9+, making
    this unnecessary.
    """

    def __get__(self, owner_obj, owner_cls):
        """Call the decorated method to convert it into a property."""
        return self.fget(owner_cls)


def isproperty(this: object) -> bool:
    """True if `this` is a Python instance property.
    
    This function is designed for use as a predicate with `inspect.getmembers`.
    """
    return inspect.isdatadescriptor(this) and isinstance(this, property)

