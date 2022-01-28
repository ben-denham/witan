import numpy as np
from numpy.testing import assert_array_equal

from witan.generator import (get_conditional_entropy_matrix,
                             get_col_entropies)


def p2e(prob):
    return -(prob * np.log2(prob) + (1 - prob) * np.log2(1 - prob))


def test_get_conditional_entropy_matrix():
    ce_matrix = get_conditional_entropy_matrix(
        rule_masks=[
            # No coverage
            np.array([0, 0, 0, 0]).astype(bool),
            # Full coverage
            np.array([1, 1, 1, 1]).astype(bool),
            # Partial coverage
            np.array([0, 0, 1, 1]).astype(bool),
            np.array([1, 1, 1, 0]).astype(bool),
        ],
        X=np.array([
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
            [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
        ]).astype(bool)
    )
    assert_array_equal(
        ce_matrix,
        np.array([
            # No coverage
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            # Full coverage
            [0, p2e(1/4), p2e(1/4), p2e(2/4), p2e(1/4), p2e(2/4), p2e(2/4), p2e(3/4),
             p2e(1/4), p2e(2/4), p2e(2/4), p2e(3/4), p2e(2/4), p2e(3/4), p2e(3/4), 0],
            # Partial coverage
            [0, 0, 0, 0, p2e(1/2), p2e(1/2), p2e(1/2), p2e(1/2),
             p2e(1/2), p2e(1/2), p2e(1/2), p2e(1/2), 0, 0, 0, 0],
            [0, p2e(1/3), p2e(1/3), p2e(2/3), p2e(1/3), p2e(2/3), p2e(2/3), 0,
             0, p2e(1/3), p2e(1/3), p2e(2/3), p2e(1/3), p2e(2/3), p2e(2/3), 0],
        ])
    )


def test_get_col_entropies():
    col_entropies = get_col_entropies(
        X=np.array([
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
            [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
        ]).astype(bool)
    )
    assert_array_equal(
        col_entropies,
        np.array([
            0, p2e(1/4), p2e(1/4), p2e(2/4), p2e(1/4), p2e(2/4), p2e(2/4), p2e(3/4),
            p2e(1/4), p2e(2/4), p2e(2/4), p2e(3/4), p2e(2/4), p2e(3/4), p2e(3/4), 0
        ])
    )
