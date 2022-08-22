import pytest

from goats.core import spelling


def test_spell_checker():
    """Test the spell-checking object."""
    words = {
        'apple': ['appple', 'appl', 'aplpe', 'Apple'],
        'dog': ['ddog', 'dg', 'odg', 'Dog'],
        'cheese': ['chheese', 'chese', 'cheees', 'Cheese'],
    }
    checker = spelling.SpellChecker(*words)
    for word, errors in words.items():
        assert checker.check(word, mode='suggest') == []
        assert checker.check(word, mode='truth')
        assert checker.check(word) is None
        for error in errors:
            assert checker.check(error, mode='suggest') == [word]
            assert not checker.check(error, mode='truth')
            with pytest.raises(spelling.SpellingError):
                checker.check(error)
    with pytest.raises(ValueError):
        checker.check('apple', mode='edit')


def test_update_words():
    """Make sure the user can update the correctly-spelled words."""
    words = ['apple', 'pear']
    checker = spelling.SpellChecker(*words)
    assert checker.words == set(words)
    checker.words |= {'pie'}
    assert checker.words == set(words) | {'pie'}

