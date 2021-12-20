from goats.common import numerical


def test_nearest():
    values = [0.1, 0.2, 0.3]
    basic = {
        0.11: (0, 0.1),
        0.15: (0, 0.1),
        0.20: (1, 0.2),
    }
    for target, (index, value) in basic.items():
        found = numerical.find_nearest(values, target)
        assert found.index == index
        assert found.value == value
    for target in [0.21, 0.25, 0.29]:
        found = numerical.find_nearest(values, target, bound='lower')
        assert found.index == 2
        assert found.value == 0.3
        found = numerical.find_nearest(values, target, bound='upper')
        assert found.index == 1
        assert found.value == 0.2
