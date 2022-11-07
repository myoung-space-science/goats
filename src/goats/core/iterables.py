import abc
import collections
import collections.abc
import inspect
from itertools import product
import functools
import numbers
import typing

from goats.core import numerical


T = typing.TypeVar('T')


def unique(*items: T) -> typing.List[T]:
    """Remove repeated items while preserving order."""
    collection = []
    for item in items:
        if item not in collection:
            collection.append(item)
    return collection


W = typing.TypeVar('W', bound=typing.Iterable)


def unwrap(
    obj: typing.Union[T, typing.Iterable[T]],
    wrap: typing.Type[W]=None,
) -> typing.Union[T, W]:
    """Remove redundant outer lists and tuples.

    This function will strip away enclosing instances of ``list`` or ``tuple``,
    as long as they contain a single item, until it finds an object of a
    different type, a ``list`` or ``tuple`` containing multiple items, or an
    empty ``list`` or ``tuple``.

    Parameters
    ----------
    obj : Any
        The object to "unwrap".

    wrap : type
        An iterable type into which to store the result. Specifying this allows
        the caller to ensure that the result is an iterable object after
        unwrapping interior iterables.

    Returns
    -------
    Any
        The element enclosed in multiple instances of ``list`` or ``tuple``, or
        a (possibly empty) ``list`` or ``tuple``.

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

    Passing a type to `wrap` ensures a result of that type:

    >>> iterables.unwrap(42, wrap=tuple)
    (42,)
    >>> iterables.unwrap(42, wrap=list)
    [42]
    >>> iterables.unwrap([42], wrap=list)
    [42]
    >>> iterables.unwrap(([(42,)],), wrap=list)
    [42]

    It works with multiple wrapped elements:

    >>> iterables.unwrap([1, 2])
    [1, 2]
    >>> iterables.unwrap([[1, 2]])
    [1, 2]
    >>> iterables.unwrap(['one', 'two'])
    ['one', 'two']
    >>> iterables.unwrap([['one', 'two']])
    ['one', 'two']

    It stops at an empty ``list`` or ``tuple``:

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
    >>> iterables.unwrap([], wrap=tuple)
    ()
    """
    seed = [obj]
    wrapped = (list, tuple)
    while isinstance(seed, wrapped) and len(seed) == 1:
        seed = seed[0]
    if wrap is not None:
        return wrap(whole(seed))
    return seed


def get_nested_element(mapping: typing.Mapping, levels: typing.Iterable):
    """Walk a mapping to get the element at the last level."""
    this = mapping[levels[0]]
    if len(levels) > 1:
        for level in levels[1:]:
            this = this[level]
    return this


def missing(this: typing.Any) -> bool:
    """True if `this` is null or empty.

    This function allows the user to programmatically test for objects that are
    logically ``False`` except for numbers equivalent to 0.
    """
    if isinstance(this, numbers.Number):
        return False
    size = getattr(this, 'size', None)
    if size is not None:
        return size == 0
    try:
        result = not bool(this)
    except ValueError:
        result = all((missing(i) for i in this))
    return result


def transpose_list(list_in: typing.List[list]) -> typing.List[list]:
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
    if not (string.startswith('[') and string.endswith(']')):
        raise TypeError(f"Can't convert {string!r} to a list") from None
    inside = string.strip('[]')
    if not inside:
        return []
    items = [item.strip(" ''") for item in inside.split(',')]
    return [numerical.cast(i) for i in items]


def naked(targets: typing.Any) -> bool:
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


def show_at_most(
    n: int,
    values: typing.Iterable[typing.Any],
    separator: str=',',
) -> str:
    """Create a string with at most `n` values."""
    seq = list(values)
    if len(seq) <= n:
        return separator.join(str(v) for v in seq)
    truncated = [*seq[:n-1], '...', seq[-1]]
    return separator.join(str(v) for v in truncated)


@typing.runtime_checkable
class Displayable(typing.Protocol):
    """Protocol for classes that use `~iterables.ReprStrMixin`."""

    @property
    @abc.abstractmethod
    def display(self) -> dict: ...


class DisplayMap:
    """An attribute mapping for string formatting."""

    def __init__(self, instance: Displayable) -> None:
        self._instance = instance

    def __getitem__(self, name: str) -> str:
        """Get the named attribute and call it if necessary."""
        attr = getattr(self._instance, self._instance.display[name])
        this = attr() if callable(attr) else attr
        return str(this)


class DisplayString(collections.UserString):
    """A list-like representation of a display string."""

    def __init__(self, seq: object) -> None:
        super().__init__(seq)
        self.separator = ' '

    def __getitem__(self, __i: typing.SupportsIndex):
        """Get a substring by index."""
        parts = self.data.split(self.separator)
        return parts[__i]

    def append(self, substring: str):
        """Append `substring` to the end of this display."""
        parts = self.data.split(self.separator)
        parts.append(substring)
        self.data = self.separator.join(parts)

    def insert(self, index: int, substring: str):
        """Insert `substring` into this display at `index`."""
        parts = self.data.split(self.separator)
        parts.insert(index, substring)
        self.data = self.separator.join(parts)

    def format_map(self, mapping: typing.Mapping[str, typing.Any]) -> str:
        return self.data.format_map(mapping)


class Display(collections.UserDict):
    """A dict-like object for string representations."""

    def __init__(self, **kwargs):
        mapping = {'__str__': '', '__repr__': '', **kwargs}
        super().__init__(mapping)

    @typing.overload
    def __getitem__(
        self,
        __k: typing.Literal['__str__', '__repr__'],
    ) -> DisplayString: ...

    @typing.overload
    def __getitem__(self, __k: str) -> str: ...

    def __getitem__(self, __k):
        return super().__getitem__(__k)

    def __setitem__(self, __k: str, __s: str) -> None:
        if __k not in {'__str__', '__repr__'}:
            raise KeyError(f"Can't set value of {__k!r}")
        self.data[__k] = DisplayString(__s)

    def register(self, *names: str, **pairs: str):
        """Set or update which attributes to show.
        
        Parameters
        ----------
        *names : iterable of strings
            Zero or more names of attributes to include in the display.

        **pairs : dict
            Zero or more key-value pairs in which the key is the name of an
            attribute in the current display and the value is the name of the
            attribute to use in its place.
        """
        for name in names:
            self.data[name] = name
        for name, alias in pairs.items():
            self.data[name] = alias


class ReprStrMixin:
    """A mixin class that provides support for `__repr__` and `__str__`."""

    _display = None

    @property
    def display(self):
        """The attributes to display for each method."""
        if self._display is None:
            self._display = Display()
        return self._display

    def __str__(self) -> str:
        """A simplified representation of this object."""
        return self._get_display('__str__')

    def __repr__(self) -> str:
        """An unambiguous representation of this object."""
        string = self._get_display('__repr__')
        module = f"{self.__module__.replace('goats.', '')}."
        name = self.__class__.__qualname__
        return f"{module}{name}({string or self})"

    def _get_display(self, method: str):
        """Helper method for `__str__` and `__repr__`."""
        target = self.display[method]
        return target.format_map(DisplayMap(self))


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
# implementing `collections.abc.MutableMapping`. However, it appears to
# significantly slow down execution.
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
    def _collection(self) -> typing.Collection:
        """The base collection."""
        try:
            this = getattr(self, self._collection_name, ())
        except AttributeError:
            indent = ' ' * len('AttributeError: ')
            raise AttributeError(
                "Cannot collect unknown attribute."
                f"\n{indent}Please pass the name of the underlying collection"
                " via the collect method"
            ) from None
        else:
            return this

    def __contains__(self, key: str) -> bool:
        """True if `key` names a member of this collection."""
        return key in self._collection

    def __len__(self) -> int:
        """The number of members in this collection."""
        return len(self._collection)

    def __iter__(self) -> typing.Iterator:
        """Iterate over members of this collection."""
        return iter(self._collection)


class MappingBase(collections.abc.Mapping):
    """A partial implementation of `collections.abc.Mapping`.

    This abstract base class is designed to serve as a basis for easily creating
    concrete implementations of `collections.abc.Mapping`. It defines simple
    implementations, based on a user-provided collection, for the abstract
    methods `__len__` and `__iter__` but leaves `__getitem__` abstract.

    Examples
    --------
    The following class implements `collections.abc.Mapping`::

        class Implemented(MappingBase):

            def __init__(self, mapping: Mapping) -> None:
                __mapping = mapping or {} super().__init__(__mapping.keys())
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
                __mapping = mapping or {} super().__init__(__mapping.keys())

    """

    def __init__(self, __collection: typing.Collection) -> None:
        """Initialize this instance with the base collection.

        Parameters
        ----------
        __collection
            Any concrete implementation of `collections.abc.Collection`. This
            attribute's implementations of the required collection methods will
            support the equivalent implementations for this mapping.
        """
        self._collection = __collection

    def __len__(self) -> int:
        """The number of members in this collection."""
        return len(self._collection)

    def __iter__(self) -> typing.Iterator:
        """Iterate over members of this collection."""
        return iter(self._collection)


class UniformMapping(MappingBase):
    """A mapping with a specified value type."""

    def __init__(
        self,
        __mapping: typing.Mapping[str, typing.Mapping],
        __type: typing.Type,
    ) -> None:
        """
        Parameters
        ----------
        __mapping
            A mapping from string to interior mapping. The items of the interior
            mapping will provide the arguments used to initialize a new instance
            of the given type.

        __type
            The type of value to return from key-based look-up.

        Examples
        --------
        Create a mapping of mappings that will provide attribute values based on
        keyword::

            m = {
                'a': {'value': +1, 'name': 'pos'},
                'b': {'value': -1, 'name': 'neg'},
            }

        Define a simple class to represent the result::

            class MyType:
                def __init__(self, value, name) -> None:
                    self.value = value
                    self.name = name
                def __str__(self) -> str:
                    return f"value={self.value:+}, name={self.name!r}"

        Create an instance and access objects by name::

            u = iterables.UniformMapping(m, MyType)
            print(u['a'])
            print(u['b'])

        This prints::

            value=+1, name='pos'
            value=-1, name='neg'

        Redefine the custom type as a named tuple::

            class MyType(typing.NamedTuple):
                value: int
                name: str

        Create an instance and access objects by name::

            u = iterables.UniformMapping(m, NamedType)
            print(u['a'])
            print(u['b'])

        This prints::

            NamedType(value=1, name='pos')
            NamedType(value=-1, name='neg')

        """
        self._mapping = __mapping
        super().__init__(self._mapping)
        self._type = __type
        self._init = None
        if issubclass(self._type, tuple):
            bases = self._type.__bases__
            if len(bases) == 1 and bases[0] == tuple:
                self._init = getattr(self._type, '_fields', None)
        if self._init is None:
            self._init = tuple(
            p for p in inspect.signature(self._type.__init__).parameters
            if p != 'self'
        )

    def __getitem__(self, key: str):
        """Create an instance of the type from the mapping."""
        if key in self:
            kwargs = {p: self._mapping[key].get(p) for p in self._init}
            return self._type(**kwargs)
        raise KeyError(key) from None


class InjectiveTypeError(TypeError):
    """The given mapping contains repeated values."""


class NonInvertibleError(TypeError):
    """The given mapping is not invertible."""


class Bijection(MappingBase):
    """An invertable mapping."""

    def __new__(cls, __mapping: typing.Mapping):
        """Check for invalid input mappings."""
        mapping = dict(__mapping)
        n_keys = len(mapping.keys())
        n_values = len(set(mapping.values()))
        if n_keys > n_values:
            raise InjectiveTypeError(
                "The given mapping is injective but not surjective."
            )
        if n_keys != n_values:
            raise NonInvertibleError(
                "The given mapping is not invertible"
                f" with {n_keys} keys and {n_values} values"
            )
        return super().__new__(cls)

    def __init__(self, __mapping: typing.Mapping) -> None:
        self._mapping = dict(__mapping)
        super().__init__(self._mapping.keys())

    def __getitem__(self, key):
        """Look up item by key."""
        return self._mapping[key]

    def invert(self):
        """Invert this mapping."""
        return type(self)({v: k for k, v in self.items()})


class ObjectRegistry(collections.abc.Mapping):
    """A class for associating metadata with abitrary objects."""
    def __init__(
        self,
        base: typing.Mapping=None,
        object_key: str='object',
    ) -> None:
        mapping = base or {}
        self._items = {
            k: v if isinstance(v, typing.Mapping) else {object_key: v}
            for k, v in mapping.items()
        }
        self._object_key = object_key
        self._init = self._items.copy()
        self._default_key_count = 1

    def __iter__(self) -> typing.Iterator:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, key: str) -> bool:
        return key in self._items

    _OT = typing.TypeVar('_OT')

    def register(
        self,
        _obj: _OT=None,
        name: str=None,
        overwrite: bool=False,
        **metadata
    ) -> _OT:
        """Register an object and any associated metadata.

        This function exists to decorate objects. Without any arguments, it will
        log the decorated object in an internal mapping, keyed by the object's
        name. The user may optionally provide key-value pairs of metadata to
        associate with the object.

        Parameters
        ----------
        name : string
            The name to use as this object's key in the internal mapping. The
            default is `None`, which causes this class to create a unique key
            based on the defined name of the object.

        overwrite : bool, default=false
            If true and there is already an object with the key given by `name`,
            overwrite that object. This keyword has no effect if `name` is
            `None`.

        **metadata
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
        obj: typing.Any,
        user: str=None,
        overwrite: bool=False
    ) -> str:
        """Get an appropriate key to associate with this object."""
        available = user not in self._items or overwrite
        if user and isinstance(user, typing.Hashable) and available:
            return user
        return self._get_default_key(obj)

    def _get_default_key(self, obj: typing.Any) -> str:
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

    def __getitem__(self, key: str) -> typing.Dict[str, typing.Any]:
        """Get an item from the object collection."""
        return self._items[key]

    def keys(self) -> typing.AbstractSet[str]:
        """Mapping keys. Defined here just to specify return type."""
        return super().keys()

    def values(self) -> typing.ValuesView[typing.Dict[str, typing.Any]]:
        """Mapping values. Defined here just to specify return type."""
        return super().values()

    def items(self) -> typing.AbstractSet[typing.Tuple[str, typing.Dict[str, typing.Any]]]:
        """Mapping items. Defined here just to specify return type."""
        return super().items()

    def copy(self) -> 'ObjectRegistry':
        """A shallow copy of this instance."""
        return ObjectRegistry(self._items.copy(), object_key=self._object_key)

    def __repr__(self) -> str:
        """An unambiguous representation of this object."""
        return f"{self.__class__.__qualname__}({self._items})"


class TableKeyError(KeyError):
    """No common key with this name."""
    def __str__(self) -> str:
        if len(self.args) > 0:
            return f"Table has no common key '{self.args[0]}'"
        return "Key not found in table"


class TableValueError(Exception):
    """An exception occurred during value-based look-up."""

    def __init__(self, value: typing.Any) -> None:
        self.value = value


class AmbiguousValueError(TableValueError):
    """Failed to find a unique entry by value."""

    def __str__(self) -> str:
        return f"No unique entry containing {self.value!r}"


class MissingValueError(TableValueError):
    """Failed to find any qualifying entries by value."""

    def __str__(self) -> str:
        return f"No entries containing {self.value!r}"


class TableRequestError(Exception):
    """An exception occurred during standard look-up."""

    def __init__(self, request: typing.Mapping) -> None:
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

    _KT = typing.TypeVar('_KT', bound=str)
    _VT = typing.TypeVar('_VT')
    _ET = typing.TypeVar('_ET', bound=typing.Mapping)

    def __init__(self, entries: typing.Collection[_ET]) -> None:
        super().__init__(entries)
        self._entries = entries
        self._keys = None

    def show(self, names: typing.Iterable[str]=None):
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
    def keys(self) -> typing.Set[_KT]:
        """All the keys common to the individual mappings."""
        if self._keys is None:
            all_keys = [list(entry.keys()) for entry in self._entries]
            self._keys = set(all_keys[0]).intersection(*all_keys[1:])
        return self._keys

    def __getitem__(self, key: _KT) -> typing.Tuple[_VT]:
        """Get all the values for a given key if it is common."""
        if key in self.keys:
            values = [entry[key] for entry in self._entries]
            return tuple(values)
        raise TableKeyError(key)

    def get(self, key: _KT, default: typing.Any=None) -> typing.Tuple[_VT]:
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
    ) -> typing.Union[_ET, typing.List[_ET]]:
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
        entries. The iterative search will raise an `AmbiguousRequestError` if
        the given key-value pairs are insufficient to determine a unique entry.

        Parameters
        ----------
        strict : bool, default=False
            Fail if any of the given key-value pairs is not in the collection.

        **request : mapping
            Key-value pairs that define the search criteria. Each key must
            appear in all table entries.

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


class Connections(collections.abc.Collection, ReprStrMixin):
    """A graph defined by a collection of weighted edges."""

    def __init__(self, base: typing.Mapping=None) -> None:
        """Initialize an instance.

        Parameters
        ----------
        base : mapping, optional
            The mapping of weighted connections from which to initialize this
            instance. Items in the mapping must have the form ``((start, end),
            weight)``.
        """
        self.connections: typing.Dict[typing.Tuple[str, str], float] = {}
        """The forward and reverse links in this graph."""
        base = base or {}
        for (start, end), weight in base.items():
            self.add_connection(start, end, weight)

    def __contains__(self, connection: typing.Tuple[str, str]):
        """True if `connection` is available."""
        return connection in self.connections

    def __len__(self) -> int:
        """The number of connections. Called for len(self)."""
        return len(self.connections)

    def __iter__(self):
        """Iterate over connections. Called for iter(self)."""
        return iter(self.connections)

    @property
    def nodes(self):
        """The distinct nodes in this graph."""
        return {n for connection in self.connections for n in connection}

    def get_adjacencies(self, node: str):
        """Retrieve the connections to this node.
        
        Parameters
        ----------
        node : string
            The key corresponding to the target node.

        Returns
        -------
        `~dict`
            A dictionary whose keys represent the nodes connected to `node` and
            whose values represent the corresponding edge weight. An empty dictionary represents a node with no connections.
        """
        return {
            end: v for (start, end), v in self.connections.items()
            if start == node
        } if node in self.nodes else {}

    def get_weight(self, start: str, end: str):
        """Retrieve the weight of this link, if possible."""
        if (start, end) in self.connections:
            return self.connections[(start, end)]
        raise KeyError(
            f"No connection between {start!r} and {end!r}"
        ) from None

    def add_connection(
        self,
        start: str,
        end: str,
        weight: float=None,
    ) -> None:
        """Add a connection (with optional weight) to the graph."""
        forward = ((start, end), weight)
        inverse = 1 / weight if weight else weight
        reverse = ((end, start), inverse)
        for edge, value in (forward, reverse):
            if edge not in self.connections:
                self.connections[edge] = value

    def __str__(self) -> str:
        """A simplified representation of this object."""
        return '\n'.join(
            f"({n0} -> {n1}): {wt}"
            for (n0, n1), wt in self.connections.items()
        )


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


class InstanceSet(abc.ABCMeta):
    """A metaclass for sets of singletons.

    Using this class as a metaclass will ensure that only one instance of the
    object class exists for a given combination of initializing arguments. Given
    a concrete implementation of `_generate_key`, this metaclass will simply
    return the appropriate existing instance instead of creating a new one.

    See https://refactoring.guru/design-patterns/singleton/python/example
    """

    _instances = {}

    def __call__(self, *args, **kwargs):
        """Ensure that only one instance of the given object exists."""
        key = self._generate_key(*args, **kwargs)
        if key not in self._instances:
            self._instances[key] = super().__call__(*args, **kwargs)
        return self._instances[key]

    @abc.abstractmethod
    def _generate_key(self, *args, **kwargs):
        """Generate a unique instance key.
        
        Concrete implementations of this method must map the given arguments to
        a valid dictionary key, which will be associated with a unique instance.
        """
        raise TypeError(
            "Can't generate unique mapping key from arguments"
        ) from None


class NothingType(Singleton):
    """An object that represents nothing in a variety of ways."""

    def __getitem__(self, index: typing.Any) -> None:
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

    def __iter__(self) -> typing.Iterable:
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

    def __repr__(self) -> str:
        """This object knows that it is nothing."""
        return "Nothing"

Nothing = NothingType()
"""A unique object that represents nothing."""


class NonStrIterable(abc.ABCMeta):
    """A type representing a non-string iterable."""

    def __instancecheck__(cls, this: typing.Any) -> bool:
        """True if `this` is not string-like and is iterable."""
        return (
            not isinstance(this, (str, bytes))
            and isinstance(this, typing.Iterable)
        )


class SeparableTypeError(TypeError):
    """Non-separable argument type."""

    def __init__(self, arg) -> None:
        self.arg = arg

    def __str__(self) -> str:
        return f"{self.arg!r} is not separable"


class whole(
    collections.abc.Collection,
    typing.Generic[T],
    ReprStrMixin,
    metaclass=NonStrIterable):
    """A collection of independent members.

    This class represents iterable collections with members that have meaning
    independent of any other members. For example, a list of numbers is
    whole whereas a string is not, despite the fact that both objects are
    iterable collections.

    The motivation for this distinction is to make it easier to treat single
    numbers and strings equivalently to iterables of numbers and strings.
    """

    def __init__(
        self,
        arg: typing.Optional[typing.Union[T, typing.Iterable[T]]],
    ) -> None:
        """Initialize a whole object from `arg`"""
        self.arg = self.parse(arg)

    @staticmethod
    def parse(
        arg: typing.Optional[typing.Union[T, typing.Iterable[T]]],
    ) -> typing.List[T]:
        """Convert `arg` into a whole object.

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
            return list(arg)

    def __iter__(self) -> typing.Iterator[T]:
        return iter(self.arg)

    def __len__(self) -> int:
        return len(self.arg)

    def __contains__(self, this: object) -> bool:
        return this in self.arg

    def __eq__(self, other: 'whole') -> bool:
        """True if two whole iterables have equal arguments."""
        if isinstance(other, whole):
            return sorted(self) == sorted(other)
        return NotImplemented

    def __str__(self) -> str:
        """A simplified representation of this object."""
        return str(self.arg)


def distribute(a, b):
    """Distribute `a` and `b` over each other.

    If both `a` and `b` are whole (see the `whole` class), this function
    will return their Cartesian product. If only `a` or `b` is whole, this
    function will pair the non-whole argument with each element of the
    whole argument. If neither is whole, this function will raise an
    error.
    """
    a_separable = isinstance(a, whole)
    b_separable = isinstance(b, whole)
    if not (a_separable or b_separable):
        raise TypeError("At least one argument must be whole")
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


def batch_replace(string: str, replacement: typing.Mapping[str, str]) -> str:
    """Replace characters in a string based on a mapping."""
    for old, new in replacement.items():
        string = string.replace(old.strip(), new)
    return string


R = typing.TypeVar('R')
def apply(
    methods: typing.Iterable[typing.Callable[..., R]],
    *args,
    **kwargs,
) -> typing.Optional[R]:
    """Apply the given methods until one returns a non-null result."""
    gen = (method(*args, **kwargs) for method in methods)
    if result := next((match for match in gen if match), None):
        return result


def allinstance(
    __args: typing.Iterable,
    __type: type,
) -> bool:
    """True if all arguments are instances of `__type`."""
    return all(isinstance(arg, __type) for arg in __args)


def extract_single(args):
    """Extract a single value from a length-1 collection."""
    try:
        n = len(args)
    except TypeError:
        n = None
    return args[0] if n == 1 else args


def class_attribute(__cls: type, name: str):
    """Collect values of a class attribute across a class hierarchy.
    
    This function builds a list of values of the named attribute by iteratively
    searching classes in the `__cls` method resolution order (MRO), all the way
    back to :builtin:``object``. If a parent class does not have the attribute,
    this function simply skips it.

    Parameters
    ----------
    __cls : type
        The class at the tip of the hierarchy.

    name : string
        The name of the attribute to collect.
    """
    ancestors = __cls.mro()[::-1]
    return [
        attr for ancestor in ancestors
        for attr in whole(getattr(ancestor, name, ()))
    ]


def pop(__x: list, default: T):
    """Pop from a list or use the default value."""
    try:
        return __x.pop(0)
    except IndexError:
        return default


def oftype(__type: typing.Type[T]):
    """Create a class that only accepts instances of `__type`."""
    def dunder_new(cls, *args: T):
        """Prevent instantiation with invalid types."""
        if not all(isinstance(arg, __type) for arg in args):
            raise TypeError(
                f"Can't instantiate {cls.__qualname__!r}"
                f" with non-{__type.__qualname__} arguments"
            ) from None
        return args[0] if len(args) == 1 else iter(args)
    return type(
        f'StrictIterable[{__type.__qualname__}]',
        (collections.abc.Iterable,),
        {'__new__': dunder_new}
    )


def hastype(
    __obj,
    __types: typing.Union[type, typing.Tuple[type, ...]],
    *wrappers: typing.Type[typing.Iterable],
    strict: bool=False,
) -> bool:
    """True if an object is a certain type or contains certain types.
    
    Parameters
    ----------
    __obj : Any
        The object to compare.

    __types : type or tuple of types
        One or more types of which the target object may be an instance.

    *wrappers : iterable type
        Zero or more iterable types of which the target object may be an
        instance. If the target object is an instance of a given wrapper type,
        this function will test whether every member of the target object is an
        instance of the given types.

    strict : bool, default=False
        If true, return ``True`` if `__obj` contains only `__type`. Otherwise,
        return ``True`` if `__obj` contains at least one of `__types`.

    Examples
    --------
    When called without wrappers, this function is identical to ``isinstance``:

    >>> iterables.hastype(1, int)
    True

    >>> iterables.hastype('s', str)
    True

    >>> iterables.hastype([1, 2], list)
    True

    Note that in these cases, `strict` is irrelevant because this function
    checks only the type of `__obj`.

    The target object contains the given type but ``list`` is not a declared
    wrapper:
    
    >>> iterables.hastype([1, 2], int)
    False
    
    Same as above, but this time ``list`` is a known wrapper:

    >>> iterables.hastype([1, 2], int, list)
    True
    
    Similar, except only ``tuple`` is declared as a wrapper:

    >>> iterables.hastype([1, 2], int, tuple)
    False

    By default, only one member of a wrapped object needs to be an instance of one of the target types:

    >>> iterables.hastype([1, 2.0], int, list)
    True

    If ``strict=True``, each member must be one of the target types:

    >>> iterables.hastype([1, 2.0], int, list, strict=True)
    False

    Multiple target types must be passed as a ``tuple``, just as when calling
    ``isinstance``:

    >>> iterables.hastype([1, 2.0], (int, float), list)
    True

    Otherwise, this function will interpret them as wrapper types:

    >>> iterables.hastype([1, 2.0], int, float, list, strict=True)
    False
    """
    if isinstance(__obj, __types):
        return True
    for wrapper in wrappers:
        if isinstance(__obj, wrapper):
            check = all if strict else any
            return check(isinstance(i, __types) for i in __obj)
    return False


G = typing.TypeVar('G')


class Guard:
    """Substitute default values for exceptions.

    This class wraps a callable object in `try/except` logic that substitutes a
    default value when calling that object raises a known exception. The default
    behavior is to substitute ``None``, but users may specify a specific
    substitution value when registering a known exception (see `~Guard.catch`).
    
    Notes
    -----
    This class was inspired by https://stackoverflow.com/a/8915613/4739101.
    """

    def __init__(self, __callable: typing.Callable[..., T]) -> None:
        self._call = __callable
        self._substitutions = {}

    def catch(self, exception: Exception, /, value: G=None):
        """Register a known exception and optional substitution value.
        
        Parameters
        ----------
        exception
            An exception class to catch when calling the guarded object.

        value, optional
            The value to return from `~Guard.call` when calling the guarded
            object raises `exception`. The default value is ``None``. There is
            one special case: Registering `exception` with ``value = ...``
            (i.e., the built-in ``Ellipsis`` object), will cause `~Guard.call` to return
            the given argument(s). See `~Guard.call` for more information about
            the form of the return value in this case.
        """
        self._substitutions[exception] = value

    def call(self, *args, **kwargs) -> typing.Union[T, G]:
        """Call the guarded object with the given arguments.
        
        Parameters
        ----------
        *args
            Positional arguments to pass to the guarded object.

        **kwargs
            Keyword arguments to pass to the guarded object.

        Returns
        -------
        The result of calling the guarded object, or an associated default
        value, or the given arguments.
        
        If no exceptions arose when calling the guarded object, this method will
        return the result of that call.

        If calling the guarded object raises a known exception and the value
        associated with that exception is not ``...`` (the built-in ``Ellipsis``
        object), this method will return the associated value.
        
        If calling the guarded object raises a known exception and the value
        associated with that exception is ``...``, this method will return the
        given arguments. The return type in this case depends on the given
        arguments. If the user passes only a single positional argument, this
        method will return that argument. If the user passes only positional
        arguments, this method will return the equivalent ``tuple``. If the user
        passes only keyword arguments, this method will return the equivalent
        ``dict``. If the user passes positional and keyword arguments, this
        method will return a ``tuple`` containing the corresponding equivalent
        ``tuple`` and ``dict``.
        
        If calling the guarded object raises an exception that is unknown to
        this instance, that exception will propagate up to the caller as usual.
        """
        try:
            return self._call(*args, **kwargs)
        except tuple(self._substitutions) as err:
            value = self._substitutions[type(err)]
            if value != Ellipsis:
                return value
            if not kwargs:
                if len(args) == 1:
                    return args[0]
                return args
            if not args:
                return kwargs
            return args, kwargs
