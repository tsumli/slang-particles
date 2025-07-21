def div_up(a: int, b: int) -> int:
    return (a + b - 1) // b


def round_up(a: int, b: int) -> int:
    assert b & (b - 1) == 0
    return a + (b - 1) & ~(b - 1)
