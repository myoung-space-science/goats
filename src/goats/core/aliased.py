import collections
import collections.abc
import typing

from goats.core import iterables


Aliases = typing.TypeVar('Aliases')
Aliases = typing.Union[str, typing.Iterable[str]]

class MappingKey(iterables.ReprStrMixin, collections.abc.Set):
    """A mapping key with associated aliases."""

    __slots__ = ('_aliases')

    _builtin = (tuple, list, set)

    @classmethod
    def supports(cls, key: str):
        """True if `key` can instantiate this class."""
        try:
            cls(key)
        except TypeError:
            return False
        return True

    def __init__(self, *a: Aliases) -> None:
        if not a:
            raise TypeError("At least one alias is required") from None
        self._aliases = self._from_iterable(a)

    @classmethod
    def _from_iterable(cls, it):
        try:
            length = len(it)
        except TypeError:
            length = None
        if length == 1 and not isinstance(it[0], str):
            return set(it[0])
        return set(it)

    def __iter__(self) -> typing.Iterator:
        return iter(self._aliases)

    def __len__(self) -> int:
        return len(self._aliases)

    def __contains__(self, key: str) -> bool:
        return key in self._aliases

    def __hash__(self) -> int:
        """Compute the hash of the underlying key set."""
        return hash(tuple(self._aliases))

    def _implement(operator):
        def method(self: 'MappingKey', other):
            return operator(self, MappingKey(other))
        def wrapper(self, other):
            result = method(self, other)
            if isinstance(result, typing.Iterable):
                return type(self)(result)
            return result
        return wrapper

    __eq__ = _implement(collections.abc.Set.__eq__)
    __and__ = _implement(collections.abc.Set.__and__)
    isdisjoint = _implement(collections.abc.Set.isdisjoint)
    __or__ = _implement(collections.abc.Set.__or__)
    __sub__ = _implement(collections.abc.Set.__sub__)
    __rsub__ = _implement(collections.abc.Set.__rsub__)
    __xor__ = _implement(collections.abc.Set.__xor__)

    def __str__(self) -> str:
        """A simplified representation of this instance."""
        return ' | '.join(str(alias) for alias in self._aliases)


_VT = typing.TypeVar('_VT')
class Mapping(collections.abc.Mapping):
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
    >>> amap = aliased.Mapping(mapping)
    >>> amap
    Mapping('a': 1, 'b | B': 2, 'c | C | c0': -9)

    The caller may access an individual item by any one (but only one) of its
    valid aliases.

    >>> amap['a']
    1
    >>> amap['b']
    2
    >>> amap['B']
    2
    >>> amap[('b', 'B')]
    ...
    KeyError: ('b', 'B')

    Aliased items are identical.

    >>> amap['b'] is amap['B']
    True

    The representation of keys, values, and items reflect the internal grouping
    by alias, but iterating over each produces de-aliased members. This behavior
    naturally supports loops and comprehensions, since access is only valid for
    individual items.

    >>> amap.keys()
    AliasedKeys(['a', 'b | B', 'c | C | c0'])
    >>> amap.values()
    AliasedValues([1, 2, -9])
    >>> amap.items()
    AliasedItems([('a', 1), ('b | B', 2), ('c | C | c0', -9)])
    >>> list(amap.keys())
    ['a', 'b', 'B', 'c', 'C', 'c0']
    >>> list(amap.values())
    [1, 2, 2, -9, -9, -9]
    >>> list(amap.items())
    [('a', 1), ('b', 2), ('B', 2), ('c', -9), ('C', -9), ('c0', -9)]
    >>> for k, v in amap.items():
    ...     print(k, amap[k], v)
    ... 
    a 1 1
    b 2 2
    B 2 2
    c -9 -9
    C -9 -9
    c0 -9 -9

    It is always possible to access the equivalent de-aliased `dict`.

    >>> amap.flat
    {'a': 1, 'b': 2, 'B': 2, 'c': -9, 'C': -9, 'c0': -9}

    Updates and deletions apply to all associated aliases.

    >>> amap['c'] = 5.6
    >>> list(amap.items())
    [('a', 1), ('b', 2), ('B', 2), ('c', 5.6), ('C', 5.6), ('c0', 5.6)]
    >>> del amap['c']
    >>> list(amap.items())
    [('a', 1), ('b', 2), ('B', 2)]

    Users may access all aliases for a given key via the
    `alias` method. Attempting to register an alias will raise a `KeyError` if
    it is already an alias for a different key.

    >>> amap.alias('b')
    MappingKey('B')
    >>> amap.alias('b', include=True)
    MappingKey('b | B')
    >>> amap.alias(b='my B')
    >>> amap.alias('b')
    MappingKey('B | my B')
    >>> amap
    Mapping('a': 1, 'b | my B | B': 2)
    >>> amap.alias(a='b')
    ...
    KeyError: "'b' is already an alias for '(B, my B)'"

    Notes
    -----
    The length of this object, as well as its keys, values, and items, is equal
    to the number of valid aliases it contains. This is consistent with the
    many-to-one nature of the mapping despite the fact that it internally stores
    aliases and values in a one-to-one mapping.

    """

    Aliasable = typing.TypeVar('Aliasable', bound=typing.Mapping)
    Aliasable = typing.Union[
        typing.Mapping[Aliases, _VT],
        typing.Mapping[str, typing.Mapping[str, typing.Any]],
    ]

    def __init__(
        self,
        mapping: typing.Union[Aliasable, 'Mapping']=None,
        aliases: str='aliases',
    ) -> None:
        """Initialize this instance."""
        self._aliased = self._build_aliased(mapping, aliases=aliases)
        self._keymap = {
            alias: key for key in self._aliased for alias in key
        }

    def _flat_keys(self) -> typing.KeysView[str]:
        """Define a flat list of all the keys in this mapping."""
        flattened = [key for keys in self._aliased.keys() for key in keys]
        return collections.abc.KeysView(flattened)

    def _build_aliased(
        self,
        mapping: typing.Mapping=None,
        aliases: str='aliases',
    ) -> dict:
        """Build the internal mapping of aliased items."""
        # Is it empty?
        if not mapping:
            return {}
        # Is it already an instance of this class?
        if isinstance(mapping, Mapping):
            return dict(mapping.items(aliased=True))
        # Does it have the form {<key>: {key: <aliases>, <k>: <value>, ...}}?
        string_keys = all(isinstance(key, str) for key in mapping)
        dict_values = all(isinstance(value, dict) for value in mapping.values())
        if string_keys and dict_values:
            return self._build_from_key(mapping, key=aliases)
        # Does it have the form {<aliased key>: <value>}?
        if all(MappingKey.supports(key) for key in mapping):
            return self._build_from_aliases(mapping)
        # Is it a built-in dictionary?
        if isinstance(mapping, dict):
            return mapping.copy()

    @property
    def flat(self) -> typing.Dict[str, _VT]:
        """Expand aliased items into a standard dictionary."""
        return {key: self[key] for key in self._flat_keys()}

    def __iter__(self) -> typing.Iterator:
        yield from self._flat_keys()

    def __len__(self) -> int:
        return len(self._flat_keys())

    def __getitem__(self, key: typing.Union[str, MappingKey]) -> _VT:
        """Look up a value by one of its keys."""
        if resolved := self._resolve(key):
            return self._aliased[resolved]
        raise KeyError(f"'{key!r}' is not a known name or alias.") from None

    def _resolve(self, key: typing.Union[MappingKey, typing.Any]):
        """Resolve `key` into an existing or new aliased key."""
        if isinstance(key, MappingKey):
            return self._look_up_key(key)
        return self._keymap.get(key)

    def _look_up_key(self, target: MappingKey):
        """Find the aliased key equivalent to `target`.
        
        Notes
        -----
        Checking ``key in self._aliased`` doesn't always work because `dict`
        look-up will first compare the key's hash value to those of existing
        keys before comparing the key itself. This may fail in the first stage
        due to the fact that `MappingKey.__hash__` uses `tuple.__hash__`, which
        depends on order, despite the fact that the second stage should pass
        because `MappingKey.__eq__` does not depend on order. See
        https://stackoverflow.com/q/327311/4739101.
        """
        try:
            found = next(
                key for key in self._aliased
                if key == target
            )
        except StopIteration:
            return
        else:
            return found

    def _build_from_aliases(
        self,
        mapping: typing.Union[Aliasable, 'Mapping'],
    ) -> typing.Dict[MappingKey, _VT]:
        """Build a `dict` that maps aliased keys to user values."""
        out = {}
        for key, value in mapping.items():
            try:
                aliased_key = next(k for k in out if key in k)
            except StopIteration:
                aliased_key = MappingKey(key)
            out[aliased_key] = value
        return out

    def _build_from_key(
        self,
        mapping: typing.Mapping[str, typing.Mapping[str, typing.Any]],
        key: str='aliases',
    ) -> typing.Dict[MappingKey, _VT]:
        """Build a `dict` with aliased keys taken from interior mappings.
        
        Parameters
        ----------
        mapping
            An object that maps string keys to interior mappings of strings to
            any type.

        key : str, default='aliases'
            The key in the interior mappings whose values to use as aliases for
            the corresponding keys in the user-provided mapping. Absence of this
            key from a given interior mapping simply means the corresponding key
            in the user-provided mapping will be the only alias for that entry.

        Examples
        --------
        Create aliased mappings from a user dictionary with the default alias
        key::

        >>> mapping = {
        ...     'a': {'aliases': ('A', 'a0'), 'n': 1, 'm': 'foo'},
        ...     'b': {'aliases': 'B', 'n': -4},
        ... }
        >>> amap = aliased.Mapping(mapping)
        >>> amap
        aliased.Mapping('a0 | A | a': {'n': 1, 'm': 'foo'}, 'b | B': {'n': -4})
        >>> amap['a']
        {'n': 1, 'm': 'foo'}

        Create aliased mappings from a user dictionary, swapping the alias key::

        >>> mapping = {
        ...     'a': {'foo': 'A', 'bar': 'a0'},
        ...     'b': {'foo': 'B', 'bar': 'b0'},
        ... }
        >>> amap = aliased.Mapping(mapping, aliases='foo')
        >>> amap
        aliased.Mapping('A | a': a0, 'b | B': b0)
        >>> amap['a']
        'a0'
        >>> amap = aliased.Mapping(mapping, aliases='bar')
        >>> amap
        aliased.Mapping('a0 | a': A, 'b | b0': B)
        >>> amap['a']
        'A'
        """
        keys = self.extract_keys(mapping, aliases=key)
        values = [
            {k: v for k, v in group.items() if k != key}
            for group in mapping.values()
        ]
        return dict(zip(keys, values))

    @classmethod
    def fromkeys(
        cls,
        mapping: typing.Mapping[str, typing.Mapping[str, typing.Any]],
        aliases: str='aliases',
        value: typing.Any=None,
    ): # How do I annotate this so it's correct for subclasses?
        """Create an aliased mapping based on another mapping's keys.

        Parameters
        ----------
        user : mapping
            Same as for `~aliases.Mapping.__init__`.

        aliases : string
            Same as for `~aliases.Mapping.__init__`.

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

        >>> mapping = {
        ...     'a': {'aliases': ('A', 'a0'), 'n': 1, 'm': 'foo'},
        ...     'b': {'aliases': 'B', 'n': -4},
        ... }
        >>> amap = aliased.Mapping.fromkeys(mapping, value=-1.0)
        >>> amap
        aliased.Mapping('a0 | a | A': -1.0, 'B | b': -1.0)
        """
        keys = cls.extract_keys(mapping, aliases=aliases)
        d = {k: value for k in keys}
        return cls(d)

    @classmethod
    def extract_keys(
        cls,
        mapping: typing.Mapping[str, typing.Mapping[str, typing.Any]],
        aliases: str='aliases',
    ) -> typing.List[MappingKey]:
        """Extract keys for use in an aliased mapping.
        
        Parameters
        ----------
        mapping
            Same as for `~aliased.Mapping.__init__`.

        aliases : string
            Same as for `~aliased.Mapping.__init__`.

        Examples
        --------
        >>> mapping = {
        ...     'a': {'aliases': ('A', 'a0'), 'n': 1, 'm': 'foo'},
        ...     'b': {'aliases': 'B', 'n': -4},
        ... }
        >>> keys = Mapping.extract_keys(mapping)
        >>> keys
        [MappingKey('a | A | a0'), MappingKey('B | b')]
        """
        if isinstance(mapping, Mapping):
            return mapping.keys(aliased=True)
        return [
            MappingKey(k) | MappingKey(v.get(aliases, ()))
            for k, v in mapping.items()
        ]

    def alias(self, *current, include=False):
        """Get the alias for an existing key."""
        if current:
            if len(current) > 1:
                raise ValueError(
                    "Can only get one aliased key at a time"
                ) from None
            key = current[0]
            if include:
                return self._resolve(key)
            return self._resolve(key) - key

    def __eq__(self, other: typing.Mapping) -> bool:
        """Define equality between this and another object."""
        if not isinstance(other, typing.Mapping):
            return NotImplemented
        if isinstance(other, Mapping):
            return self.items() == other.items()
        return dict(self.items()) == dict(other.items())

    def __or__(self, other: 'Mapping'):
        """Merge this aliased mapping with another."""
        items = dict((*self.items(aliased=True), *other.items(aliased=True)))
        return type(self)(items)

    def __str__(self) -> str:
        """A simplified representation of this instance."""
        return ', '.join(
            f"'{g}': {v!r}"
            for g, v in zip(self._aliased.keys(), self._aliased.values())
        )

    def __repr__(self) -> str:
        """An unambiguous representation of this object."""
        module = f"{self.__module__.replace('goats.', '')}."
        name = self.__class__.__qualname__
        return f"{module}{name}({self})"

    def keys(self, aliased: bool=False):
        """A view on this instance's keys."""
        return KeysView(self, aliased=aliased)

    def values(self, aliased: bool=False):
        """A view on this instance's values."""
        return ValuesView(self, aliased=aliased)

    def items(self, aliased: bool=False):
        """A view on this instance's key-value pairs."""
        return ItemsView(self, aliased=aliased)

    def copy(self):
        """Create a shallow copy of this instance."""
        return type(self)(self._aliased)


class MappingView(iterables.ReprStrMixin, collections.abc.MappingView):
    """Base class for views of aliased mappings."""

    __slots__ = ('_mapping', '_keys')

    def __init__(self, mapping: Mapping, aliased: bool=False) -> None:
        super().__init__(mapping)
        aliases = mapping._aliased.keys()
        self._keys = aliases if aliased else mapping._flat_keys()
        self._mapping = mapping

    def __len__(self):
        """Called for len(self)."""
        return len(self._keys)

    def __str__(self):
        """A simplified representation of this object."""
        return str(list(self))


class KeysView(MappingView, collections.abc.KeysView):
    """A view on the keys of an aliased mapping."""

    def __iter__(self):
        """Iterate over aliased mapping keys."""
        yield from self._keys


class ValuesView(MappingView, collections.abc.ValuesView):
    """A view on the values of an aliased mapping."""

    def __iter__(self):
        """Iterate over aliased mapping values."""
        for key in self._keys:
            yield self._mapping[key]


class ItemsView(MappingView, collections.abc.ItemsView):
    """A view on the key-value pairs of an aliased mapping."""

    def __iter__(self):
        """Iterate over aliased mapping items."""
        for key in self._keys:
            yield (key, self._mapping[key])


class MutableMapping(Mapping, collections.abc.MutableMapping):
    """A mutable version of `Mapping`.
    
    Parameters
    ----------
    mapping : mapping
        An object that maps strings or iterables of strings to values of any
        type.

    Examples
    --------
    These examples build on the examples shown in `Mapping`.

    >>> mutable = iterables.MutableMapping(aliased)

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
    MappingKey('B')
    >>> mutable.alias('b', include=True)
    MappingKey('b | B')
    >>> mutable.alias(b='my B')
    >>> mutable.alias('b')
    MappingKey('B | my B')
    >>> mutable
    Mapping('a': 1, 'b | my B | B': 2)
    >>> mutable.alias(a='b')
    ...
    KeyError: "'b' is already an alias for '(B, my B)'"

    Notes
    -----
    See note on lengths at `~Mapping`.
    """

    def __setitem__(self, key: str, value: _VT):
        """Assign a value to `key` and its aliases."""
        resolved = self._resolve(key) or MappingKey(key)
        self._aliased[resolved] = value
        self._refresh()

    def __delitem__(self, key: str):
        """Remove the item corresponding to `key`."""
        resolved = self._resolve(key)
        if not resolved:
            raise KeyError(f"'{key!r}' is not a known name or alias.") from None
        del self._aliased[resolved]
        self._refresh()

    def _refresh(self):
        """Perform common tasks after setting or deleting an item."""
        self._keymap = {
            alias: key for key in self._aliased for alias in key
        }

    def alias(self, *current: str, include=False, **new: str):
        """Get the alias for an existing key or register new ones."""
        if current and new:
            raise TypeError(
                "Can't get and set aliases at the same time"
            ) from None
        if current:
            if len(current) > 1:
                raise ValueError(
                    "Can only get one aliased key at a time"
                ) from None
            key = current[0]
            if include:
                return self._resolve(key)
            return self._resolve(key) - key
        for key, alias in new.items():
            if alias:
                if alias in self._flat_keys():
                    raise self._not_available(alias) from None
                updated = self._resolve(key) | alias
                self._aliased[updated] = self[key]
                del self[key]

    def _not_available(self, key: str) -> typing.NoReturn:
        """True if this key is not currently in use."""
        aliases = self.alias(key)
        this = ", ".join(f'{a}' for a in aliases)
        if len(aliases) > 1:
            this = f"({this})"
        return KeyError(f"'{key}' is already an alias for {this!r}")


class NameMap(iterables.MappingBase):
    """A mapping from aliases to canonical names.
    
    Notes
    -----
    See note on lengths at `~Mapping`.
    """

    AliasDefinitions = typing.TypeVar('AliasDefinitions')
    AliasDefinitions = typing.Union[
        typing.Iterable[typing.Iterable[str]],
        typing.Mapping[str, typing.Iterable[str]],
        typing.Mapping[str, typing.Mapping[str, typing.Iterable[str]]],
    ]

    AliasReferences = typing.TypeVar('AliasReferences')
    AliasReferences = typing.Union[
        typing.Iterable[str],
        typing.Mapping[str, typing.Any],
    ]

    def __init__(
        self,
        defs: AliasDefinitions,
        refs: AliasReferences=None,
        key: str='aliases',
    ) -> None:
        names = self._get_names(defs, refs)
        self._mapping = self._build_mapping(names, defs, key)
        super().__init__(self._mapping.keys())
        self._init = {'refs': refs, 'defs': defs, 'key': key}

    RT = typing.TypeVar('RT')
    RT = typing.Union[
        typing.Iterable[str],
        typing.Iterable[typing.Iterable[str]],
    ]
    def _get_names(
        self,
        defs: AliasDefinitions,
        refs: AliasReferences=None,
    ) -> RT:
        """Create an iterable of canonical names, if possible."""
        if isinstance(refs, typing.Mapping):
            return refs.keys()
        if isinstance(refs, typing.Iterable):
            return refs
        if isinstance(defs, typing.Mapping):
            return defs.keys()
        raise TypeError(
            f"Can't create name map from {defs!r} and {refs!r}"
        ) from None

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
        namemap = MutableMapping(identity)
        updates = self._get_aliases(names, aliases, key)
        namemap.alias(**updates)
        return Mapping(namemap)

    def _get_aliases(self, names, defs: AliasDefinitions, key):
        """Determine the appropriate aliases for each canonical name."""
        # Mapping <: Iterable, so we need to check Mapping first.
        if isinstance(defs, typing.Mapping):
            # There are two allowed types of Mapping:
            # 1) Mapping[str, Mapping[str, Iterable[str]]]
            # 2) Mapping[str, Iterable[str]]
            
            # Make sure the keys are all strings.
            if any(k for k in defs if not isinstance(k, str)):
                raise TypeError("All aliases must be strings") from None
            # Again, we need to check Mapping values before Iterable values.
            if all(isinstance(v, typing.Mapping) for v in defs.values()):
                return {
                    name: tuple(v.get(key, ()))
                    for name, v in defs.items()
                }
            if all(isinstance(v, typing.Iterable) for v in defs.values()):
                return {name: tuple(aliases) for name, aliases in defs.items()}
        # Alias definitions are in a non-mapping iterable. We may want to
        # further check that each member of `defs` is itself an iterable of
        # strings.
        only_iterables = all(isinstance(d, typing.Iterable) for d in defs)
        if isinstance(defs, typing.Iterable) and only_iterables:
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
            for aliases, name in self._mapping.items(aliased=True)
        }
        self._mapping = Mapping(inverted)
        return self

    def __getitem__(self, key: str):
        """Get the canonical name for `key`."""
        if key in self._mapping:
            return self._mapping[key]
        raise KeyError(key)

    def __str__(self) -> str:
        """A simplified representation of this object."""
        items = {f"{k}: '{v}'" for k, v in self._mapping.items(aliased=True)}
        return ', '.join(items)

    def __repr__(self) -> str:
        """An unambiguous representation of this object."""
        return f"{self.__class__.__qualname__}({self})"

    def keys(self, aliased: bool=False):
        """A view on this object's aliased keys."""
        return Mapping.keys(self._mapping, aliased=aliased)

    def values(self, aliased: bool=False):
        """A view on this object's aliased values."""
        return Mapping.values(self._mapping, aliased=aliased)

    def items(self, aliased: bool=False):
        """A view on this object's aliased key-value pairs."""
        return Mapping.items(self._mapping, aliased=aliased)

    def copy(self):
        """Make a shallow copy of this object."""
        return type(self)(**self._init)


class KeyMap(iterables.MappingBase):
    """A collection that associates common aliases."""

    def __init__(self, __keys: typing.Iterable[Aliases]) -> None:
        """
        Parameters
        ----------
        __keys : iterable
            An iterable collection of associated keys. Each key may be a string
            or an iterable of strings (including instances of `~MappingKey`).
        """
        self._aliased = [MappingKey(key) for key in __keys]
        self._flat = [key for alias in self._aliased for key in alias]
        super().__init__(self._flat)

    def __getitem__(self, key: str) -> MappingKey:
        """Look up aliases for key."""
        try:
            found = next(entry for entry in self._aliased if key in entry)
        except StopIteration as err:
            raise KeyError(f"{key!r} not found") from err
        else:
            return found


