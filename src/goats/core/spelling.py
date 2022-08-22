import typing


class SpellingError(Exception):
    """It seems like the user just misspelled the variable's name."""
    def __init__(
        self,
        name: str,
        suggested: typing.Union[str, typing.List[str]],
    ) -> None:
        self.name = name
        self.suggested = suggested

    def __str__(self) -> str:
        return f"Could not find '{self.name}'. {self.suggestion}"

    @property
    def suggestion(self) -> str:
        """The formatted spelling suggestion."""
        if isinstance(self.suggested, str):
            return f"Did you mean '{self.suggested}'?"
        if len(self.suggested) == 1:
            return f"Did you mean '{self.suggested[0]}'?"
        return f"Did you mean one of {self.suggested}?"


class SpellChecker:
    """A simple spell-checker for named quantities.
    
    This is based on https://norvig.com/spell-correct.html
    """
    def __init__(self, *words: str) -> None:
        self.words = set(words)
        letters = 'abcdefghijklmnopqrstuvwxyz'
        self.letters = letters + letters.upper()

    def check(self, name: str, mode: str=None):
        """Check the spelling of `name` based on known words."""
        suggestions = self._check(name)
        if mode == 'suggest':
            return suggestions
        if mode == 'truth':
            return not suggestions
        if mode is not None:
            raise ValueError(f"Unknown mode {mode!r}") from None
        if suggestions:
            raise SpellingError(name, suggestions)

    def _check(self, name: str):
        """Internal helper for `~SpellChecker.check`."""
        if name in self.words:
            return []
        edits = list(self.edits(name))
        suggestions = self.get_suggestions(edits)
        if not suggestions:
            edits = [w for edit in edits for w in list(self.edits(edit))]
            suggestions = self.get_suggestions(edits)
        return suggestions

    def get_suggestions(self, edits: list):
        """Get a batch of suggested words based on `edits`."""
        return [n for n in self.words if n in edits]

    def known(self, words: typing.Iterable[str]) -> set:
        """The subset of `words` that is in the list of known words."""
        return {word for word in words if word in self.words}

    def edits(self, word: str) -> typing.Set[typing.List[str]]:
        """All edits that are one edit away from `word`."""
        return set(
            self.deletes(word)
            + self.transposes(word)
            + self.replaces(word)
            + self.inserts(word)
        )

    def splits(self, word: str) -> typing.List[str]:
        """The strings made by splitting `word` at each pair of letters."""
        return [
            (word[:i], word[i:])
            for i in range(len(word))
        ]

    def deletes(self, word: str) -> typing.List[str]:
        """All words produced by deleting one letter from `word`."""
        return [
            left + right[1:]
            for left, right in self.splits(word) if right
        ]

    def transposes(self, word: str) -> typing.List[str]:
        """All words produced by transposing one pair of letters in `word`."""
        return [
            left + right[1] + right[0] + right[2:]
            for left, right in self.splits(word) if len(right) > 1
        ]

    def replaces(self, word: str) -> typing.List[str]:
        """All words produced by replacing one letter in `word`."""
        return [
            left + c + right[1:] 
            for left, right in self.splits(word) if right for c in self.letters
        ]

    def inserts(self, word: str) -> typing.List[str]:
        """All words produced by inserting one letter into `word`."""
        return [
            left + c + right
            for left, right in self.splits(word) for c in self.letters
        ]


