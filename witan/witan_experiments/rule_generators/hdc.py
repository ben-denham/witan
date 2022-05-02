import numpy as np
import pandas as pd
from time import time
from typing import NamedTuple, List, Dict, Any

from .base import Rule, RulesetGenerator
from ..utils import log


def get_binary_entropies(probs: np.ndarray) -> np.ndarray:
    """Given an array of probabilities of binary variables, return their entropies."""
    x = probs
    y = 1 - probs
    # Use values of zero where the input value is zero (which we can't take the log of).
    log_x = np.log2(x, out=np.zeros(x.shape), where=(x > 0.0))
    log_y = np.log2(y, out=np.zeros(y.shape), where=(y > 0.0))
    entropies = -(x * log_x) - (y * log_y)
    assert entropies.shape == probs.shape
    return entropies


class EntropyStats(NamedTuple):
    col_pos_sums: np.ndarray
    col_pos_probs: np.ndarray
    col_entropies: np.ndarray
    cond_pos_entropy_matrix: np.ndarray
    cond_neg_entropy_matrix: np.ndarray
    cond_entropy_matrix: np.ndarray
    info_gain_matrix: np.ndarray
    norm_info_gain_matrix: np.ndarray
    mean_norm_info_gains: np.ndarray


def get_entropy_stats(X: np.array) -> EntropyStats:
    """
    Calculate entropy and info-gain matrices for the given binary feature matrix.
    """
    col_pos_sums = np.sum(X, axis=0)
    col_neg_sums = X.shape[0] - col_pos_sums
    # col_pos_probs[i] = P(Xi)
    col_pos_probs = col_pos_sums / X.shape[0]
    col_neg_probs = 1 - col_pos_probs
    # col_entropies[i] = H(Xi)
    col_entropies = get_binary_entropies(col_pos_probs)

    cond_pos_entropy_matrix = np.ones((X.shape[1], X.shape[1]))
    cond_neg_entropy_matrix = np.ones((X.shape[1], X.shape[1]))
    for cond_idx in range(X.shape[1]):
        cond_pos_sum = col_pos_sums[cond_idx]
        cond_neg_sum = col_neg_sums[cond_idx]
        if (cond_pos_sum == 0) or (cond_neg_sum == 0):
            # A constant condition does not reduce entropies
            cond_pos_entropy_matrix[:, cond_idx] = col_entropies
            cond_neg_entropy_matrix[:, cond_idx] = col_entropies
        else:
            cond_pos = X[:, cond_idx].reshape((X.shape[0], 1))
            # cond_pos_probs[i] = P(Xi | cond_pos)
            cond_pos_probs = np.sum(cond_pos & X, axis=0) / cond_pos_sum
            # cond_neg_probs[i] = P(Xi | ~cond_pos)
            cond_neg_probs = np.sum(~cond_pos & X, axis=0) / cond_neg_sum
            # cond_pos_entropy_matrix[i, j] = H(Xi | Xj=1)
            cond_pos_entropy_matrix[:, cond_idx] = get_binary_entropies(cond_pos_probs)
            # cond_neg_entropy_matrix[i, j] = H(Xi | Xj=0)
            cond_neg_entropy_matrix[:, cond_idx] = get_binary_entropies(cond_neg_probs)

    # cond_entropy_matrix[i, j] = H(Xi | Xj) = P(Xj=1)H(Xi | Xj=1) + P(Xj=0)H(Xi | Xj=1)
    cond_entropy_matrix = (
        # P(cond_pos) * H(Xi | cond_pos)
        (col_pos_probs * cond_pos_entropy_matrix) +
        # P(cond_neg) * H(Xi | cond_neg)
        (col_neg_probs * cond_neg_entropy_matrix)
    )
    # info_gain_matrix[i, j] = IG(Xi, Xj) = H(Xi) - H(Xi | Xj)
    info_gain_matrix = (col_entropies - cond_entropy_matrix.T).T
    # normaliser_matrix[i, j] = H(Xi) + H(Xj)
    tiled_col_entropies = np.tile(col_entropies, (col_entropies.shape[0], 1))
    normaliser_matrix = tiled_col_entropies.T + tiled_col_entropies
    # norm_info_gain_matrix[i, j] = NIG(Xi, Xj) = IG(Xi, Xj) / (H(Xi) + H(Xj))
    norm_info_gain_matrix = np.divide(info_gain_matrix, normaliser_matrix,
                                      out=np.zeros(info_gain_matrix.shape),
                                      where=normaliser_matrix > 0)
    # NOTE: The mean for a feature i is a mean over the gains for i
    # conditioned on other features j. Because norm_info_gain_matrix
    # is symmetric, swapping i and j doesn't make a difference, but it
    # would make a difference for a matrix of gain ratios.
    # mean_norm_info_gains[i] = (sum(NIG(Xi, Xj) forall j != i)) / (|J| - 1)
    mean_norm_info_gains = (
        (np.sum(norm_info_gain_matrix, axis=1) - np.diagonal(norm_info_gain_matrix)) /
        (X.shape[1] - 1)
    )

    return EntropyStats(
        col_pos_sums=col_pos_sums,
        col_pos_probs=col_pos_probs,
        col_entropies=col_entropies,
        cond_pos_entropy_matrix=cond_pos_entropy_matrix,
        cond_neg_entropy_matrix=cond_neg_entropy_matrix,
        cond_entropy_matrix=cond_entropy_matrix,
        info_gain_matrix=info_gain_matrix,
        norm_info_gain_matrix=norm_info_gain_matrix,
        mean_norm_info_gains=mean_norm_info_gains,
    )


class HdcRG(RulesetGenerator):
    """Generates rules from clusters produced with hierarchical divisive
    clustering, based on the MGR algorithm of:

    Qin, H., Ma, X., Herawan, T., & Zain, J. M. (2014). MGR: An
    information theory based hierarchical divisive clustering
    algorithm for categorical data. Knowledge-Based Systems, 67,
    401-411.

    with the mean-normalised-info-gain and continual splitting of the
    largest bipartition improvements proposed in:

    Wei, W., Liang, J., Guo, X., Song, P., & Sun,
    Y. (2019). Hierarchical division clustering framework for
    categorical data. Neurocomputing, 341, 118-134.

    mag: Minimum accuracy gap above a random classifier for the simulated user to label a rule
    mcr: Minimum coverage of candidate features

    """

    def __init__(self, *,
                 mag: float = 0.2,
                 mcr: float = 0.02):
        self.min_accuracy_gap = self.mag = mag
        self.min_cov_ratio = self.mcr = mcr

    def get_min_accuracy(self, classes: np.array) -> float:
        """Return the minimum accuracy for a simulated user to label a
        rule."""
        if self.min_accuracy_gap is None:
            return 0
        chance_accuracy = 1 / classes.shape[0]
        return chance_accuracy + self.min_accuracy_gap

    def generate_ruleset(self,
                         classes: np.array,
                         train_binary_features: pd.DataFrame,
                         train_y: pd.Series,
                         seed_rules: List[Rule],
                         rngseed: int,
                         interaction_count: int) -> Dict[str, Any]:
        min_accuracy = self.get_min_accuracy(classes)
        train_binary_features = train_binary_features.loc[:, train_binary_features.mean() >= self.min_cov_ratio]

        C = train_binary_features.to_numpy()
        current_C_mask = pd.Series(True, index=train_binary_features.index)
        current_C_predicate = ''
        interactions = []
        for i in range(interaction_count):
            log(f'Starting MGR interaction {i}')

            if C.shape[0] == 0:
                rule_useful = False
                rule = None
                rule_series = None
                interaction_wall_seconds = 0.0
            else:
                interaction_start_time = time()
                entropy_stats = get_entropy_stats(C)
                # Condition on the feature with the highest mean normalised info-gain
                best_cond_idx = np.argmax(entropy_stats.mean_norm_info_gains)
                if entropy_stats.col_pos_sums[best_cond_idx] == C.shape[0]:
                    # If there are only instances with the positive
                    # condition, then this positive cluster will be
                    # the final cluster.
                    negate_cond = False
                elif entropy_stats.col_pos_sums[best_cond_idx] == 0:
                    # If there are only instances with the negative
                    # condition, then this negative cluster will be
                    # the final cluster.
                    negate_cond = True
                else:
                    # Negate the condition if the positive condition
                    # is larger (we negate the rule's condition so
                    # that we continue to split the larger cluster)
                    negate_cond = entropy_stats.col_pos_probs[best_cond_idx] > 0.5
                cond_mask_of_C = C[:, best_cond_idx]
                if negate_cond:
                    cond_mask_of_C = ~cond_mask_of_C
                # Remove rows matched by the selected condition from C
                C = C[~cond_mask_of_C]

                interaction_end_time = time()
                interaction_wall_seconds = interaction_end_time - interaction_start_time

                cond_feature_name = train_binary_features.columns[best_cond_idx]
                cond_mask = train_binary_features[cond_feature_name]
                if negate_cond:
                    cond_mask = ~cond_mask

                # Construct the rule for the selected condition
                rule_mask = current_C_mask & cond_mask
                rule_covered_y = train_y[rule_mask]
                target_class = rule_covered_y.mode().iloc[0]
                rule_accuracy = (rule_covered_y == target_class).mean()
                rule_useful = rule_accuracy >= min_accuracy
                rule = Rule(
                    target_class=target_class,
                    predicate_key=f'{current_C_predicate}{"~" if negate_cond else ""}{cond_feature_name}',
                )
                rule_series = (pd.Series(np.nan, index=train_binary_features.index)
                               .mask(rule_mask, rule.target_class)
                               .rename(rule))

                # Update the current_C_mask and current_C_predicate to
                # reflect that subsequent conditions will be found by
                # searching within rows not matched by the rule's condition.
                current_C_mask = current_C_mask & ~cond_mask
                current_C_predicate += f'{"" if negate_cond else "~"}{cond_feature_name}&'

            log(f'Finished MGR interaction {i} for {"" if rule_useful else "non-"}useful rule: {rule}')
            interactions.append({
                'rule_useful': rule_useful,
                'rule_series': rule_series,
                'wall_seconds': interaction_wall_seconds,
            })

        return {
            'interactions': interactions,
            'rule_index': train_binary_features.index,
        }

    def get_rule_df(self, ruleset: Dict[str, Any], interaction_count: int) -> pd.DataFrame:
        useful_rules = [
            interaction['rule_series']
            for interaction in ruleset['interactions'][:interaction_count]
            if interaction['rule_useful']
        ]
        if useful_rules:
            return pd.concat(useful_rules, axis=1)
        else:
            return pd.DataFrame(index=ruleset['rule_index'])

    def get_elapsed_wall_seconds(self, ruleset: Dict[str, Any], interaction_count: int) -> float:
        elapsed_wall_seconds = 0
        for interaction in ruleset['interactions'][:interaction_count]:
            elapsed_wall_seconds += interaction['wall_seconds']
        return elapsed_wall_seconds

    def supports_partial_interactions(self) -> bool:
        return True
