from goats.core import spelling


def test_spell_checker():
    """Test the spell-checking object."""
    words = {
        'apple': ['appple', 'appl', 'aplpe', 'Apple'],
        'dog': ['ddog', 'dg', 'odg', 'Dog'],
        'cheese': ['chheese', 'chese', 'cheees', 'Cheese'],
    }
    checker = spelling.SpellChecker(words)
    for word, errors in words.items():
        assert not checker.misspelled(word)
        for error in errors:
            assert checker.misspelled(error)

