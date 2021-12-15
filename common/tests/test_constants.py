from goats.common import constants


def test_constants():
    """Test the constants mapping."""
    for key, data in constants.metadata.items():
        for system in ('mks', 'cgs'):
            d = data[system]
            mapping = constants.Constants(system)
            c = mapping[key]
            assert float(c) == d['value']
            assert c.unit == d['unit']
