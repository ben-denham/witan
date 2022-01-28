from dataclasses import dataclass
import numpy as np
import pandas as pd
from time import time
from typing import List, Optional, Any, Dict, cast

from witan import Witan
from witan.generator import WitanRule
from witan.taxonomy import TaxonomyNode, OrCondition, FeatureCondition, prune_taxonomy

from .base import Rule, RulesetGenerator
from ..utils import log


def build_seed_taxonomy(seed_rules: List[Rule]) -> TaxonomyNode:
    """Convert a list of (single feature) rules into a Witan-compatible taxonomy."""
    taxonomy = TaxonomyNode()
    for seed_rule in seed_rules:
        assert seed_rule.metadata['single_feature_rule']
        taxonomy.children.append(TaxonomyNode(
            parent=taxonomy,
            target_class=seed_rule.target_class,
            condition=OrCondition(or_conditions=(
                FeatureCondition(feature_name=seed_rule.metadata['feature_name']),
            )),
            useful=True,
        ))
    return taxonomy


def taxonomy_to_rule_df(taxonomy: TaxonomyNode,
                        bow_df: pd.DataFrame, *,
                        labelled_only: bool = True,
                        depth: int = 0) -> pd.DataFrame:
    """Convert a Witan taxonomy to a rule_df. If labelled_only=True, then
    ignore rules that do not have a target class."""
    rule_df = pd.concat([
        # Include empty df to handle case of no children
        pd.DataFrame([], index=bow_df.index),
        *[
            taxonomy_to_rule_df(
                child_node, bow_df,
                labelled_only=labelled_only,
                depth=(depth + 1),
            )
            for child_node in taxonomy.children
        ]
    ], axis=1)

    include_rule = (
        (not labelled_only or taxonomy.target_class) and
        # Exclude not-useful rules.
        (taxonomy.useful is None or taxonomy.useful)
    )
    if include_rule:
        rule = Rule(
            target_class=taxonomy.target_class,
            predicate_key=taxonomy.serialized,
            metadata={
                'depth': depth,
                'node': taxonomy,
            }
        )
        rule_series = pd.Series(None, index=bow_df.index)
        rule_series[taxonomy.get_mask(bow_df)] = taxonomy.target_class
        # Add rule as first column in rule_df
        rule_df.insert(0, rule, rule_series)

    return rule_df


@dataclass
class WitanInteraction:
    """Result of a single Witan iteration for a user interaction."""
    wall_seconds: float
    added_node: Optional[TaxonomyNode]
    rule_series: Optional[pd.DataFrame] = None


@dataclass
class WitanRuleset:
    """Result of running the WitanRG."""
    seed_taxonomy: TaxonomyNode
    seed_rule_df: pd.DataFrame
    init_wall_seconds: float
    taxonomy: TaxonomyNode
    interactions: List[WitanInteraction]


class WitanRG(RulesetGenerator):
    """Generates rules with the Witan algorithm.

    mag: Minimum accuracy gap above a random classifier for the simulated user to label a rule
    mcr: Minimum coverage of generated nodes and analysed features
    a: If True, enable genereration of conjunctive/AND rules
    o: Maximum number of features to include in an OR condition; unlimited if None
    f: If True, provide continuous user feedback of "approving" or "rejecting" each rule
    ge: The exponent applied to information gain, > 1 prefers higher gain over higher coverage
    """

    def __init__(self,
                 mag: float = 0.2,
                 mcr: float = 0.02,
                 a: bool = True,
                 o: Optional[int] = None,
                 f: bool = False,
                 ge: int = 2):
        self.min_accuracy_gap = self.mag = mag
        self.min_cov_ratio = self.mcr = mcr
        self.use_ands = self.a = a
        self.max_ors = self.o = o
        self.feedback = self.f = f
        self.gain_exp = self.ge = ge

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
                         interaction_count: int) -> WitanRuleset:
        assert train_binary_features.shape[0] == train_y.shape[0]

        min_accuracy = self.get_min_accuracy(classes)

        seed_taxonomy = build_seed_taxonomy(seed_rules)
        seed_rule_df = taxonomy_to_rule_df(seed_taxonomy, train_binary_features, labelled_only=True)

        def oracle_func(rule_mask: np.ndarray) -> Optional[str]:
            """Witan can query this "simulated user" function to label rules."""
            rule_covered_y = train_y[rule_mask]
            target_class = rule_covered_y.mode().iloc[0]
            rule_accuracy = (rule_covered_y == target_class).mean()
            useful = rule_accuracy >= min_accuracy
            if useful:
                return target_class
            else:
                return None

        init_start_time = time()
        witan = Witan()
        witan.init(train_binary_features,
                   initial_taxonomy=seed_taxonomy,
                   min_lf_coverage_ratio=self.min_cov_ratio,
                   candidate_parent_node=(None if self.use_ands else seed_taxonomy),
                   max_ors=self.max_ors,
                   gain_exponent=self.gain_exp)
        init_end_time = time()

        interactions = []
        for i in range(interaction_count):
            log(f'Running Witan iteration: {i + 1}')
            result = witan.increment_taxonomy(
                oracle_func=(oracle_func if self.feedback else None),
            )
            node = result.node

            if not self.feedback and node is not None:
                node.target_class = oracle_func(cast(WitanRule, result.rule).mask)

            interactions.append(WitanInteraction(
                added_node=node,
                wall_seconds=result.wall_seconds,
            ))

            if node is not None:
                log(f'Rule added: {node.serialized} (useful={node.useful}, class={node.target_class})')
            else:
                log(f'No rule added')

        # Add rule_series to each interaction.
        rule_df = taxonomy_to_rule_df(witan.taxonomy, train_binary_features, labelled_only=True)
        node_id_to_rule = {
            id(rule.metadata['node']): rule
            for rule in rule_df.columns
        }
        for interaction in interactions:
            if interaction.added_node is not None:
                rule = node_id_to_rule.get(id(interaction.added_node))
                if rule is not None:
                    interaction.rule_series = rule_df[rule]

        return WitanRuleset(
            seed_taxonomy=seed_taxonomy,
            seed_rule_df=seed_rule_df,
            init_wall_seconds=(init_end_time - init_start_time),
            taxonomy=witan.taxonomy,
            interactions=interactions,
        )

    def get_rule_df(self, ruleset: WitanRuleset, interaction_count: int) -> pd.DataFrame:
        interactions_rule_series = [
            interaction.rule_series
            for interaction in ruleset.interactions[:interaction_count]
            if interaction.rule_series is not None
        ]
        return pd.concat([
            ruleset.seed_rule_df,
            *interactions_rule_series,
        ], axis=1)

    def get_elapsed_wall_seconds(self, ruleset: WitanRuleset, interaction_count: int) -> float:
        elapsed_wall_seconds = ruleset.init_wall_seconds
        for interaction in ruleset.interactions[:interaction_count]:
            elapsed_wall_seconds += interaction.wall_seconds
        return elapsed_wall_seconds

    def get_extras(self, ruleset: WitanRuleset, interaction_count: int) -> Dict[str, Any]:
        interaction_nodes = [
            interaction.added_node
            for interaction in ruleset.interactions[:interaction_count]
            if interaction.added_node is not None
        ]
        return {
            'seed_taxonomy': ruleset.seed_taxonomy,
            'full_taxonomy': ruleset.taxonomy,
            'taxonomy': prune_taxonomy(ruleset.taxonomy, keep_nodes=interaction_nodes),
        }

    def supports_partial_interactions(self) -> bool:
        return True
