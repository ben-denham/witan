import numpy as np

from witan_experiments.rule_generators.hdc import get_entropy_stats


def test_get_entropy_stats():
    # Test case based on: MGR: An information theory based
    # hierarchical divisive clustering algorithm for categorical data
    # (Qin et al., 2014); Table 2 and 3
    X = np.array([
        # Experience, IT, Mathematics, Programming, Statistics, FULL_TRUE, FULL_FALSE
        [False, True, True, True, True, True, False],
        [False, True, True, True, True, True, False],
        [False, False, True, True, True, True, False],
        [False, False, True, True, False, True, False],
        [False, False, False, False, False, True, False],
        [False, False, False, False, False, True, False],
        [True, True, False, False, False, True, False],
        [True, True, False, False, True, True, False],
    ])
    entropy_stats = get_entropy_stats(X)

    np.testing.assert_array_equal(
        entropy_stats.col_pos_sums,
        np.array([2, 4, 4, 4, 4, 8, 0]),
    )
    np.testing.assert_array_almost_equal(
        entropy_stats.col_pos_probs,
        np.array([0.25, 0.5, 0.5, 0.5, 0.5, 1.0, 0.0]),
    )
    np.testing.assert_array_almost_equal(
        entropy_stats.col_entropies,
        np.array([0.811278, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]),
    )
    np.testing.assert_array_almost_equal(
        entropy_stats.cond_pos_entropy_matrix,
        np.array([
            [0, 1, 0, 0, 0.811278, 0.811278, 0.811278],
            [0, 0, 1, 1, 0.811278, 1, 1],
            [0, 1, 0, 0, 0.811278, 1, 1],
            [0, 1, 0, 0, 0.811278, 1, 1],
            [1, 0.811278, 0.811278, 0.811278, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]),
    )
    np.testing.assert_array_almost_equal(
        entropy_stats.cond_neg_entropy_matrix,
        np.array([
            [0, 0, 1, 1, 0.811278, 0.811278, 0.811278],
            [0.918296, 0, 1, 1, 0.811278, 1, 1],
            [0.918296, 1, 0, 0, 0.811278, 1, 1],
            [0.918296, 1, 0, 0, 0.811278, 1, 1],
            [1, 0.811278, 0.811278, 0.811278, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]),
    )
    np.testing.assert_array_almost_equal(
        entropy_stats.cond_entropy_matrix,
        np.array([
            [0, 0.5, 0.5, 0.5, 0.811278, 0.811278, 0.811278],
            [0.688722, 0, 1, 1, 0.811278, 1, 1],
            [0.688722, 1, 0, 0, 0.811278, 1, 1],
            [0.688722, 1, 0, 0, 0.811278, 1, 1],
            [1, 0.811278, 0.811278, 0.811278, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]),
    )
    np.testing.assert_array_almost_equal(
        entropy_stats.info_gain_matrix,
        np.array([
            [0.811278, 0.311278, 0.311278, 0.311278, 0, 0, 0],
            [0.311278, 1, 0, 0, 0.188722, 0, 0],
            [0.311278, 0, 1, 1, 0.188722, 0, 0],
            [0.311278, 0, 1, 1, 0.188722, 0, 0],
            [0, 0.188722, 0.188722, 0.188722, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]),
    )
    np.testing.assert_array_equal(entropy_stats.info_gain_matrix,
                                  entropy_stats.info_gain_matrix.T)
    np.testing.assert_array_almost_equal(
        entropy_stats.norm_info_gain_matrix,
        np.array([
            [0.5, 0.171855, 0.171855, 0.171855, 0, 0, 0],
            [0.171855, 0.5, 0, 0, 0.094361, 0, 0],
            [0.171855, 0, 0.5, 0.5, 0.094361, 0, 0],
            [0.171855, 0, 0.5, 0.5, 0.094361, 0, 0],
            [0, 0.094361, 0.094361, 0.094361, 0.5, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]),
    )
    np.testing.assert_array_equal(entropy_stats.norm_info_gain_matrix,
                                  entropy_stats.norm_info_gain_matrix.T)
    np.testing.assert_array_almost_equal(
        entropy_stats.mean_norm_info_gains,
        np.array([0.0859275, 0.044369, 0.127703, 0.127703, 0.04718, 0, 0]),
    )
