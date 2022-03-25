# Changelog

<!--next-version-placeholder-->

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