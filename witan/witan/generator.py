from dataclasses import dataclass
import numpy as np
import pandas as pd
from time import time
from threading import Lock
from typing import Optional, Set, Tuple, List, Callable, cast

from .taxonomy import (TaxonomyNode, Condition, UnaryCondition,
                       OrCondition, FeatureCondition, NotCondition,
                       get_columns_in_taxonomy)


def get_binary_entropies(probs: np.ndarray) -> np.ndarray:
    """Given an array of probabilities of binary variables, return their
    entropies."""
    x = probs
    y = 1 - probs
    # Use values of zero where the input value is zero (which we can't take the log of).
    log_x = np.log2(x, out=np.zeros(x.shape), where=(x > 0.0))
    log_y = np.log2(y, out=np.zeros(y.shape), where=(y > 0.0))
    entropies = -(x * log_x) - (y * log_y)
    assert entropies.shape == probs.shape
    return entropies


def get_col_entropies(X: np.ndarray) -> np.ndarray:
    """Return the unconditional entropy of each column (as a binary variable)."""
    col_probs = np.sum(X, axis=0) / X.shape[0]
    return get_binary_entropies(col_probs)


def get_conditional_entropy_matrix(rule_masks: List[np.ndarray],
                                   X: np.ndarray) -> np.ndarray:
    """Return an array where each row contains the entropies
    of columns in X conditional on a given rule mask."""
    probs = np.ones((len(rule_masks), X.shape[1]))
    for i, rule_mask in enumerate(rule_masks):
        rule_mask_sum = np.sum(rule_mask)
        if rule_mask_sum == 0:
            probs[i, :] = 0
        else:
            rule_mask = rule_mask.reshape((rule_mask.shape[0], 1))
            probs[i, :] = np.sum(rule_mask & X, axis=0) / rule_mask_sum
    ce_matrix = get_binary_entropies(probs)
    assert ce_matrix.shape == (len(rule_masks), X.shape[1])
    return ce_matrix


@dataclass(eq=False)
class WitanRule:
    """Representation of a candidate LF/rule for Witan, with fields to
    cache expensive computations."""
    mask: np.ndarray
    parent_mask: np.ndarray
    parent_node: Optional[TaxonomyNode]
    condition: Condition
    h_vector: Optional[np.ndarray] = None
    useful: Optional[bool] = None

    @property
    def name(self) -> str:
        """User-friendly representation of this rule."""
        result = self.condition.serialize()
        if not isinstance(self.condition, UnaryCondition):
            result = f'({result})'
        if self.parent_node:
            parent_serialized = self.parent_node.serialized
            if len(parent_serialized) > 0:
                result = f'{parent_serialized}&{result}'
        return result

    def get_all_columns(self) -> Set[str]:
        """Return all columns used to evaluate this rule's condition."""
        columns = self.condition.get_all_columns()
        if self.parent_node:
            columns = columns.union(self.parent_node.get_all_columns())
        return columns

    def get_or_conditions(self) -> Tuple[Condition]:
        """Return all top-level OR-ed conditions of this rule."""
        if isinstance(self.condition, OrCondition):
            return self.condition.or_conditions
        else:
            return (self.condition,)


@dataclass(eq=False, frozen=True)
class WitanIncrementResult:
    rule: Optional[WitanRule]
    node: Optional[TaxonomyNode]
    wall_seconds: float


class Witan():
    """Implementation of the Witan algorithm, with parameters:

    * initial_taxonomy: Can be specified to provide "seed" LFs
    * candidate_parent_node: If specified, all generated nodes must be children of candidate_parent_node
    * min_lf_coverage_ratio: Minimum coverage of generated nodes and analysed features
    * max_ors: Maximum number of features to include in an OR condition; unlimited if None
    * use_nots: Whether to generate nodes that can be NOT conditions of features
    * gain_exponent: The exponent applied to information gain, > 1 prefers higher gain over higher coverage
    * max_candidate_ratio: Factor of initial feature-based candidates by which to allow the candidate set to grow; unlimited if None
    * auto_class_prefix: If specified, generated nodes will be assigned an automatic target_class with the given prefix
    """

    def init(self,
             bow_df: pd.DataFrame,
             *,
             initial_taxonomy: Optional[TaxonomyNode] = None,
             candidate_parent_node: Optional[TaxonomyNode] = None,
             min_lf_coverage_ratio: float = 0.02,
             max_ors: Optional[int] = None,
             use_nots: bool = False,
             gain_exponent: float = 2,
             max_candidate_ratio: Optional[float] = 1,
             auto_class_prefix: Optional[str] = None):
        self.taxonomy = TaxonomyNode() if initial_taxonomy is None else initial_taxonomy
        self.min_lf_coverage = (min_lf_coverage_ratio * bow_df.shape[0])
        self.candidate_parent_node = candidate_parent_node
        self.max_ors = max_ors
        self.use_nots = use_nots
        self.gain_exponent = gain_exponent
        self.max_candidate_ratio = max_candidate_ratio
        self.auto_class_prefix = auto_class_prefix

        self.increment_lock = Lock()

        min_lf_coverage_mask = bow_df.sum(axis=0) >= self.min_lf_coverage
        # Do not let min_lf_coverage exclude columns included in the seed taxonomy.
        seed_columns = get_columns_in_taxonomy(self.taxonomy, bow_df.columns)
        if len(seed_columns) > 0:
            min_lf_coverage_mask[seed_columns] = True
        self.bow_df = bow_df.loc[:, min_lf_coverage_mask]
        self.columns = self.bow_df.columns
        self.X = np.asarray(self.bow_df.to_numpy() > 0)
        self.col_entropies = get_col_entropies(self.X)

        if self.max_candidate_ratio is None:
            self.max_candidates = None
        else:
            self.max_candidates = round(self.max_candidate_ratio * self.X.shape[1])
            # Using "nots" will initially create two candidates per feature column.
            if self.use_nots:
                self.max_candidates = self.max_candidates * 2

        self.H = np.tile(self.col_entropies, (self.X.shape[0], 1))
        self.reviewed_rules: List[WitanRule] = []
        self.rule_candidates: List[WitanRule] = []

        self.update_for_new_node(self.taxonomy)

    def update_for_new_node(self, root_node: TaxonomyNode) -> None:
        """Update the state of Witan for a new node (and it's children) added
        to the taxonomy."""
        rule_node_pairs = []

        def build_node_rules(node: TaxonomyNode,
                             parent_mask: Optional[np.ndarray] = None) -> None:
            parent_mask = (node.get_parent_mask(self.bow_df)
                           if parent_mask is None else parent_mask)
            rule = WitanRule(
                mask=(parent_mask & node.get_condition_mask(self.bow_df)),
                parent_mask=parent_mask,
                parent_node=node.parent,
                condition=node.condition,
                useful=node.useful,
            )
            rule_node_pairs.append((rule, node))
            for child in node.children:
                build_node_rules(child, parent_mask=parent_mask)

        build_node_rules(root_node)
        self.compute_rule_entropy([rule for rule, _ in rule_node_pairs])
        for rule, node in rule_node_pairs:
            self.update_for_added_rule(rule, node)

    def update_for_added_rule(self,
                              new_rule: WitanRule,
                              new_node: TaxonomyNode) -> None:
        """Update the state of Witan for an added rule."""
        # Update the reviewed_rules used for weighting.
        if new_rule.useful is not None:
            self.reviewed_rules.append(new_rule)

        # We update H even for non-useful rules, as it helps us avoid new
        # rules with the same information that was deemed un-useful.
        self.H[new_rule.mask] = np.minimum(self.H[new_rule.mask], new_rule.h_vector)

        # Always remove this rule from the candidates.
        if new_rule in self.rule_candidates:
            self.rule_candidates.remove(new_rule)

        # For useful rules (or those without feedback), update the
        # candidate rules.
        if new_rule.useful is None or new_rule.useful:

            # Remove sibling candidates that have a subset of this
            # rule's top-level OR conditions.
            to_remove = []
            new_rule_or_conditions = set(new_rule.get_or_conditions())
            for rule in self.rule_candidates:
                is_sibling = (rule.parent_node == new_rule.parent_node)
                is_subset = set(rule.get_or_conditions()).issubset(new_rule_or_conditions)
                if is_sibling and is_subset:
                    to_remove.append(rule)
            for rule in to_remove:
                self.rule_candidates.remove(rule)

            # Optionally restrict candidates to only children of the
            # candidate_parent_node.
            if (
                    (self.candidate_parent_node is None) or
                    (new_node == self.candidate_parent_node)
            ):
                # Add nested candidate rules with & conditions on an
                # additional column.
                excluded_columns = new_node.get_all_columns()
                new_candidates = []
                for j, column in enumerate(self.columns):
                    # Don't re-condition a column already in the new rule.
                    if column in excluded_columns:
                        continue
                    new_candidates.append(WitanRule(
                        mask=(new_rule.mask & self.X[:, j]),
                        parent_mask=new_rule.mask,
                        parent_node=new_node,
                        condition=OrCondition(
                            or_conditions=(
                                FeatureCondition(feature_name=column),
                            ),
                        ),
                    ))
                    if self.use_nots:
                        new_candidates.append(WitanRule(
                            mask=(new_rule.mask & ~self.X[:, j]),
                            parent_mask=new_rule.mask,
                            parent_node=new_node,
                            condition=OrCondition(
                                or_conditions=(
                                    NotCondition(FeatureCondition(feature_name=column)),
                                ),
                            ),
                        ))
                # Remove any candidates that don't meet our min_lf_coverage.
                new_candidates = [rule for rule in new_candidates
                                  if rule.mask.sum() >= self.min_lf_coverage]
                self.compute_rule_entropy(new_candidates)
                self.rule_candidates += new_candidates

    def compute_rule_entropy(self, rules: List[WitanRule]) -> None:
        """Compute the conditional entropies for each rule and cache in the rule objects."""
        rule_masks = [rule.mask for rule in rules]
        ce_matrix = get_conditional_entropy_matrix(rule_masks, self.X)
        for i, rule in enumerate(rules):
            rule.h_vector = ce_matrix[i]

    def get_w(self) -> np.ndarray:
        """Compute the current weights vector w based on reviewed_rules."""
        reviewed_ig_sum = np.zeros(self.columns.shape)
        for rule in self.reviewed_rules:
            if not rule.useful:
                continue
            rule_ig = np.maximum(0, self.col_entropies - rule.h_vector)
            ig_sum = rule_ig.sum()
            if ig_sum > 0:
                # Normalise
                rule_ig = rule_ig / ig_sum
                # Add to sum
                reviewed_ig_sum += rule_ig

        if reviewed_ig_sum.sum() == 0:
            return np.ones(self.X.shape[1])
        return reviewed_ig_sum

    def get_rules_feature_gains(self, rule_candidates: List[WitanRule]) -> np.ndarray:
        """Return an array where each row contains the unweighted gains for a
        rule candidate on each feature."""
        rules_feature_gains = np.zeros((len(rule_candidates), self.X.shape[1]))
        for i, rule in enumerate(rule_candidates):
            feature_gains = np.sum(
                np.maximum(0, self.H[rule.mask] - rule.h_vector) ** self.gain_exponent,
                axis=0,
            )
            included_column_mask = ~self.columns.isin(rule.get_all_columns())
            rules_feature_gains[i, :] = feature_gains * included_column_mask
        return rules_feature_gains

    def extend_rule(self, init_rule_idx: int, *,
                    rule_candidates: List[WitanRule],
                    rules_feature_gains: np.ndarray) -> WitanRule:
        """Attempt to improve a rule by extending it as an OR condition."""
        best_rule = rule_candidates[init_rule_idx]
        best_rule_feature_gains = rules_feature_gains[init_rule_idx]

        sibling_rules_mask = np.array([
            (rule.parent_node == best_rule.parent_node) and (rule != best_rule)
            for rule in rule_candidates
        ])
        sibling_rules = np.array(rule_candidates)[sibling_rules_mask]
        sibling_rules_feature_gains = rules_feature_gains[sibling_rules_mask]

        if sibling_rules.shape[0] == 0:
            return best_rule

        while True:
            # Limit allowed number of features in each OR condition.
            if self.max_ors is not None and len(best_rule.get_or_conditions()) >= self.max_ors:
                break

            # Weight based on the gains of best_rule.
            w = np.maximum(0, self.col_entropies - best_rule.h_vector)
            # Multiply each row by weights determined from best_rule, and sum the rows.
            sibling_gains = np.matmul(sibling_rules_feature_gains, w)
            best_sibling_idx = np.argmax(sibling_gains)
            best_sibling_rule = sibling_rules[best_sibling_idx]

            new_rule = WitanRule(
                mask=(best_rule.mask | best_sibling_rule.mask),
                parent_mask=best_rule.parent_mask,
                parent_node=best_rule.parent_node,
                condition=OrCondition(
                    or_conditions=cast(Tuple[Condition], (
                        *best_rule.get_or_conditions(),
                        *best_sibling_rule.get_or_conditions(),
                    )),
                )
            )
            self.compute_rule_entropy([new_rule])
            new_rule_feature_gains = self.get_rules_feature_gains([new_rule])[0, :]

            # Stop iterating when the new rule does not exceed the utility of best_rule.
            if np.sum(best_rule_feature_gains * w) >= np.sum(new_rule_feature_gains * w):
                break

            best_rule = new_rule
            best_rule_feature_gains = new_rule_feature_gains
            sibling_rules_feature_gains[best_sibling_idx, :] = 0.0
        return best_rule

    def increment_taxonomy(self, oracle_func: Optional[Callable[[np.ndarray], Optional[str]]] = None) -> WitanIncrementResult:
        """Generates a new node and adds it to the taxonomy. If oracle_func is
        provided, it will be called to get the target_class for the approved
        nodes, and None for rejected nodes."""
        # Lock so that multiple threads don't call
        # increment_taxonomy() at the same time.
        self.increment_lock.acquire()
        try:
            start_time = time()

            best_rule = None
            if len(self.rule_candidates) > 1:
                rules_feature_gains = self.get_rules_feature_gains(self.rule_candidates)
                # Multiply each row by the weights, and sum the rows.
                candidate_utilities = np.matmul(rules_feature_gains, self.get_w())

                # Limit the number of candidates after computing utilities.
                if self.max_candidates is not None and len(self.rule_candidates) > self.max_candidates:
                    top_candidate_idxs = np.argpartition(candidate_utilities, -self.max_candidates)[-self.max_candidates:]
                    self.rule_candidates = np.array(self.rule_candidates)[top_candidate_idxs].tolist()
                    rules_feature_gains = rules_feature_gains[top_candidate_idxs, :]
                    candidate_utilities = candidate_utilities[top_candidate_idxs]

                init_rule_idx = np.argmax(candidate_utilities)
                best_rule = self.extend_rule(init_rule_idx,
                                             rule_candidates=self.rule_candidates,
                                             rules_feature_gains=rules_feature_gains)

            if best_rule is None:
                end_time = time()
                return WitanIncrementResult(
                    rule=None,
                    node=None,
                    wall_seconds=(end_time - start_time),
                )

            oracle_start_time = time()
            target_class = None
            if oracle_func is not None:
                oracle_target_class = oracle_func(best_rule.mask)
                if oracle_target_class is None:
                    best_rule.useful = False
                else:
                    best_rule.useful = True
                    target_class = oracle_target_class
            oracle_end_time = time()

            best_node = TaxonomyNode(
                parent=best_rule.parent_node,
                condition=best_rule.condition,
                useful=best_rule.useful,
                target_class=target_class,
            )
            cast(TaxonomyNode, best_node.parent).children.append(best_node)
            if self.auto_class_prefix:
                best_node.target_class = f'{self.auto_class_prefix}{best_node.serialized}'
            self.update_for_added_rule(best_rule, best_node)

            end_time = time()

            return WitanIncrementResult(
                rule=best_rule,
                node=best_node,
                wall_seconds=((end_time - start_time) - (oracle_end_time - oracle_start_time)),
            )
        finally:
            self.increment_lock.release()
