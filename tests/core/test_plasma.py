from goats.core import plasma

def test_species():
    """Test the object that represents a single plasma species."""
    cases = [
        {'symbol': 'H+'},
        {'mass': 1.00797, 'charge': 1.0},
    ]
    for case in cases:
        species = plasma.Species(**case)
        assert species.symbol == 'H+'
        assert all(float(m) == 1.00797 for m in (species.mass, species.m))
        assert all(float(q) == 1.0 for q in (species.charge, species.q))
