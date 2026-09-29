"""Small, model-independent diagnostics for one-dimensional statistic scans."""

import numpy as np


def threshold_intervals(x, statistic, level):
    """Connected intervals where the piecewise-linear scan is <= level.

    ``x`` is an increasing grid with at least two points. Endpoints at the
    grid edges are marked: these are limits of the scanned range, not inferred
    crossings. All components are retained, including isolated equality points.
    The function makes no convexity or unique-minimum assumption.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(statistic, dtype=float) - level
    intervals = []
    for i in range(len(x) - 1):
        left, right = x[i], x[i + 1]
        below_left, below_right = y[i] <= 0, y[i + 1] <= 0
        if not (below_left or below_right):
            continue
        if below_left != below_right:
            crossing = left - y[i] * (right - left) / (y[i + 1] - y[i])
            if below_left:
                right = crossing
            else:
                left = crossing
        if intervals and left <= intervals[-1][1]:
            intervals[-1][1] = right
        else:
            intervals.append([left, right])
    return [dict(lower=float(a), upper=float(b),
                 lower_at_edge=bool(a == x[0]), upper_at_edge=bool(b == x[-1]))
            for a, b in intervals]
