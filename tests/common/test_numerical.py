from goats.common import numerical

def test_nearest():
    sequence = [0.1, 0.2, 0.3]
    basic = {
        0.11: (0, 0.1),
        0.15: (0, 0.1),
        0.20: (1, 0.2),
    }
    for value, expected in basic.items():
        assert numerical.find_nearest(sequence, value) == expected
    for value in [0.21, 0.25, 0.29]:
        found = numerical.find_nearest(sequence, value, constraint='lower')
        assert found == (2, 0.3)
        found = numerical.find_nearest(sequence, value, constraint='upper')
        assert found == (1, 0.2)
