# Changelog

## NEXT

## v0.2.7

- Fix bug for EPREM point-observer class.

## v0.2.6

- Fix additional 'egrid' dimension bug.

## v0.2.5

- Update `eprem.Axes` to handle 1D or 2D EPREM 'egrid'.

## v0.2.4

- Fix indexing bug in `index.Quantity`.
- Update eprem/test_dataset.py to check additional axis-indexing cases.

## v0.2.3

- Update `numpy.simps` to `numpy.simpson`.
- Define `observing.Context.keys` and refactor container-like methods.
- Significant updates to core/aliased.py, including replacing `KeyMap` with `Groups`, renaming `MappingKey` to `Group`, and improving type annotations.
- Fix bug in eprem/runtime.py CLI.

## v0.2.2

- Improve slice handling in `variable.Array`.

## v0.2.1

- Allow user to specify output destination when running eprem/runtime.py as a script.
- Add full and simplified logo images.
- Convert use of `constant.Assumption` and `constant.Option` to `operational.Argument`.
- Refactor eprem/runtime.py CLI and add options.

## v0.2.0

(Accidentally omitted)

## v0.1.1

- Fix bug in interpolation caused by target value being outside the bounds of the coordinate reference array.

## v0.1.0

- Restrict NumPy version to "<1.23" due to bug when running ufuncs on metadata attributes.
- Remove names from `axis.Quantity` and `index.Quantity`.
- Extract `axis.indexer` helper function.
- Remove `axis.Interface`. Observers should define their own interface to dataset axes. If it becomes clear that there is a lot of commonality among axis interfaces, we will consider defining a new base class.
- Rename classes in core/observing.py:
  - `Implementation` -> `Observable`
  - `Result` -> `Observation`
- Further improve EPREM observer path handling.
- Rename package to Generalized Observing Application Tool Suite.

## v0.0.36 (07Dec2022)

- Refactor observing framework. Observers must define a concrete implementation of the abstract base class `observing.Context`.
- Allow user to create EPREM observers without explicit paths; source and config paths are required at the time of observation.
- Rename 'axes' to 'dimensions' in regard to variable quantities.
- Remove 'name' from physical quantities.
- Redefine `variable.Quantity` to reduce inheritance.
- Deprecate core/observable.py; merge `observable.Quantity` into `observing.Implementation`.
- Redefine interface to EPREM runtime constants.
- Fix subscription bug in `variable.Array`.
- Redefine `variable.Quantity` representation to improve readability.

## v0.0.35 (11Nov2022)

- End use of `goats.Environment`.
- Fix additional environment-related bugs.

## v.0.0.34 (11Nov2022)

- Deprecate use of `Environment` for EPREM.
- Future releases may entirely remove `goats.Environment` and related `.ini` files.

## v0.0.33 (10Nov2022)

- Add Jupyter notebook with examples.
- Incorporate examples into README.md

## v0.0.32 (10Nov2022)

- Fix bug in `observer.Interface` spell-checking.
- Add dimension-related quantities to the EPREM observer's formally observable
  quantities, in order to support more composed observable quantities.

## v0.0.31 (08Nov2022)

- Fix bug related to where `eprem.Observer` searches for data files.
- Convert each array-like default EPREM parameter value to a string representation of a list.

## v0.0.30 (07Nov2022)

- Convert `observer.Interface` metric system to a read-only property.
- Redefine `index.Quantity`.
- Define `index.Array`: subclass of `index.Quantity`.
- Add `axis.Quantity.reference` property: instance of `index.Array`.
- Add EPREM observer axis-related properties

## v0.0.29 (26Sep2022)

- Ensure that `observed.Quantity.array` returns a `numpy.ndarray`.
- Add properties based on names of observable quantities to `observer.Interface`.
- Extend interpolation support.

## v0.0.28 (22Sep2022)

- Modify (partially revert) observer interface and observing API
- Pass names of unobservable quantities to observer interface
- Provide access to unobservable quantities as variable quantities
- Support changing observer's data source after initialization
- Reimplement quantity unit updates to return a new instance
- Merge index.py functionality into axis.py
- Add a 'mode' feature to spell checking
- Extract symbolic.py from algebraic.py
- Add features to symbolic (previously algebraic) Term and Expression
- Make significant updates to metric quantities, dimensions, units, and conversions
- Create a Scalar when initializing Array with a single value

## v0.0.27 (20Aug2022)

- Develop new observer interface and observing API.
- Improve string representation of observable quantity.
- Add support for a default EPREM data directory.

## v0.0.26 (16Aug2022)

- Refactor core interface to observable quantities.
- Update EPREM observer based on new interface.

## v0.0.25 (12Jul2022)

- Bug fixes.

## v0.0.24 (11Jul2022)

- New hierachy of measurable quantities determined by metadata attributes.
- New mixins for metadata attributes.
- Refactor initialization of measurable quantities.

## v0.0.23 (06Jul2022)

- Redefine most of datatypes.py based on new measurable.py.
- Create metadata.py to support operations on metadata for measurable quantities.
- Overhaul object hierarchy in measurables.py and rename to measurable.py.
- Change `base.Observable`: add `update` method, redefine `observe`, and remove `reset`.
- Initialize `datatypes.Variable` with zero or more aliased names.
- Extract `datatypes.Array` from `datatypes.Variable`.
- Extract measurables.py from quantities.py
- Rename quantities.py to metric.py
- Rename objects in metric.py, including renaming `MetricSystem` to `System`.


## v0.0.22 (29Mar2022)

- Add `goats.__init__.Environment.path`.
- Add `pytest` and `pytest-cov` to development dependencies.
- Accept multiple mappings of observables in `base.Observer`.
- Implement EPREM observer parameter access via `__getitem__`.
- Instances of `quantities.Measured` are always `True`.
- Create `Assumption` and `Option` in `eprem.parameters`.

## v0.0.21 (28Mar2022)

- Define `iotools.search`.
- Bug fix: Implement search for `goats.ini`.
- Add default `goats.ini`.

## v0.0.20 (25Mar2022)

- Define `eprem.observables.MKS`; use only MKS throughout.
- Create `core.datatypes` module for `Variable`, `Axes`, etc.
- Update how `datatypes.Variable` handles `numpy` ufunc.
- Implement `core.datasets.Variable.__measure__`.
- Change default `Variable.name` to empty string.
- Add `Variable.shape` property.

## v0.0.19 (16Mar2022)

- Define new viewer classes for `eprem.datasets.Dataset`.
- Fix slicing bug in `quantities.Measurement.__getitem__`.
- Let integral flux function accept multiple minimum energies.
- Multi-valued assumptions in derived observables.
- Create `iterables.InstanceSet`.
- Redefine handling of mass-number quantities.
- Rename `common` subpackage to `core`.
- Create `conftest.py` for `core` tests.
- Redefine `core.datasets.Dataset.axes`.
- Create `DatasetVariable` and `DatasetAxes` in `core.datasets`.
- Convert `variables` and `axes` to aliased mappings in `core.datasets.DatasetView`.
- Rename `aliased.AliasMap` to `KeyMap`.
- Create `core.datasets.Variables` class.
- Move `core.quantities.Variable` class to `datasets`.
- Add script for creating test datasets.
- Use MKS for all variables; let the user convert if necessary.
- Add new axis-related objects to `core.datasets`.
- Create `core.datasets.Dataset`.
- Handle `numpy` arrays in `iterables.missing`.
- Restructure some elements of `eprem` subpackage.

## v0.0.18 (01Mar2022)

- Fix CGS bugs in unit conversions.

## v0.0.17 (01Mar2022)

- Support strings in `algebra.Term.__eq__`.
- Update `quantities.Unit.__new__`.
- Update and improve unit conversions.

## v0.0.16 (25Feb2022)

- Update conversions in `eprem.datasets.standardize`.
- Update named units in `quantities.py`.
- Make `algebra.Expression` an immutable sequence.
- Implement `algebra.Term.__int__`.
- Overhaul unit conversions `quantities.py`.
- Redefine `quantities.Measurement` as `quantities.Vector` subclass.
- Allow user to cast a single-valued `Measurement` to `int` or `float`.
- Update `physical.PlasmaSpecies`.

## v0.0.15 (11Feb2022)

- Use `eprem.__init__.py` for the public API.
- New observable methods: `bv_mag`, `v_para`, and `v_perp`.

## v0.0.14 (10Feb2022)

- Fix scaling of flux and derived quantities.

## v0.0.13 (10Feb2022)

- Support N-D arrays in `numerical.find_nearest`.
- Do not squeeze values in `numerical.find_nearest`.
- Improve efficiency of `eprem.observables` interpolation.

## v0.0.12 (08Feb2022)

- Register scalar, vector, and variable pytest marks.
- Re-implement objects in the `quantities.Quantified` hierarchy with `__new__`.
- Make `quantities.Measured` concrete and implement `unit`.
- Redefine `quantities.Variable` operators and array access.
- Re-implement `Variable` as direct subclass of `Measured`.

## v0.0.11 (01Feb2022)

- Remove `quantities.Variable` as parent class of `base.Observation`.
- Define `Observation.axes` property from indices keys
- Define `Observation.parameters` property from assumptions keys
- Define `Observation.__getitem__` to get context or array values
- Define `Observation.unit` method (fluent property)
- Redefine `Variable.__init__` to make the class idempotent.

## v0.0.10 (31Jan2022)

- Make `base.Observation` a subclass of `quantities.Variable`.

## v0.0.9 (28Jan2022)

- Fix an interpolation bug.
- Rename `Measured.to` method to `Measured.with_unit` for clarity.
- Add `Variable.name` attribute.

## v0.0.8 (27Jan2022)

- Create `Compound` observable.

## v0.0.7 (27Jan2022)

- Define `observe` and `reset` methods in `base.Observable`.
- Add Environment to `goats.__init__.py`.

## v0.0.6 (26Jan2022)

- Create `common.base` module.
- Merge _BASETYPES_H.json and _CONFIGURATION_C.json into parameters.json.
- Create `iterables.Bijection` for use in `parameters.ConfigurationC`.
- Create `goats.ini` and `goats.ini.template`.

## v0.0.5 (18Jan2022)

- Create `common.aliased` module.
- Update and extend aliased mapping creation methods.
- Update aliased mapping views.

## v0.0.4 (14Jan2022)

- Read default parameter arguments from C code.
- Major updates to organization and interface of EPREM datasets.

## v0.0.3 (05Jan2022)

- Add /.vscode to .gitignore
- Reduce reliance on `iterables.CollectionMixin` class.
- Refactor `iterables.AliasedMapping` to improve efficiency.
- Fix interpolation bugs in integral flux.
- Create `iterables.AliasMap`.
- Simplify `functions.Functions`.
- Toggle parsing whitespace as multiplication in `algebra.Expression`.
- Let `algebra.Term` use 'a/b' instead of '(a/b)' when constant.
- Suppress printing '1' in algebraic terms and expressions.
- Add support for multiplication and equality operators to `algebra.Term`.
- Improve string formatting of algebraic terms and expressions.
- Add pytest marks for term, component, & expression.
- Allow 1-, 2-, or 3-argument forms when initializing `algebra.Term`.
- Update `quantities.UnitTerm` based on updates to `algebra.Term`.
- Refactor and extend algebraic parsing tools and corresponding tests.


## v0.0.2 (20Dec2021)

- Add simple cache to EPREM observables factory.
- Update observer objects into sub-package `__init__` modules.
- Force `Variable` to return a `Scalar` when subscription produces a single value.
- Fix bugs in EPREM functions, including key-based access and function algorithms.
- Redefine general axis-indexing objects to reduce complexity.
- Handle explicit species in EPREM energy-indexing object.
- Add new tests.

## v0.0.1 (16Dec2021)

- First release of `goats`!