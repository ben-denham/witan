import numpy as np
import pandas as pd

from witan_experiments.utils.rules import Rule
from witan_experiments.rule_generators.iws import (
    IWSBinaryRG,
    IWSDistinctRG,
    IWSMultiRG,
)


def test_binary_get_raw_lf_feature_rows():
    classes = np.array(['foo', 'bar'])
    lfs = pd.DataFrame({
        Rule('bar', '1'): [0, 1, 1, 0, 0],
        Rule('foo', '2'): [1, 1, 0, 0, 0],
        Rule('bar', '3'): [0, 0, 0, 1, 1],
    }).astype(bool)
    np.testing.assert_array_equal(
        IWSBinaryRG().get_raw_lf_feature_rows(classes, lfs),
        np.array([
            [0, -1, -1, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 0, 0, -1, -1],
        ]),
    )


def test_multi_get_raw_lf_feature_rows():
    classes = np.array(['foo', 'bar', 'bin'])
    lfs = pd.DataFrame({
        Rule('bin', '1'): [1, 1, 0, 0, 0],
        Rule('bar', '2'): [0, 1, 1, 0, 0],
        Rule('foo', '3'): [0, 0, 1, 1, 0],
        Rule('bar', '4'): [0, 0, 0, 1, 1],
    }).astype(bool)
    np.testing.assert_array_equal(
        IWSDistinctRG().get_raw_lf_feature_rows(classes, lfs),
        np.array([
            [1, 1, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 1, 1],
        ]),
    )
    np.testing.assert_array_equal(
        IWSMultiRG().get_raw_lf_feature_rows(classes, lfs).toarray(),
        np.array([
            [-1, -1, 0, 0, 0, -1, -1, 0, 0, 0, 1, 1, 0, 0, 0],
            [0, -1, -1, 0, 0, 0, 1, 1, 0, 0, 0, -1, -1, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, -1, -1, 0, 0, 0, -1, -1, 0],
            [0, 0, 0, -1, -1, 0, 0, 0, 1, 1, 0, 0, 0, -1, -1],
        ]),
    )
