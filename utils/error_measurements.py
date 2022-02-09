def relative_error(expected, actual):
    if expected == 0.0:
        if actual == 0.0:
            return 0
        else:
            return actual
    return abs(expected - actual)/abs(expected)
