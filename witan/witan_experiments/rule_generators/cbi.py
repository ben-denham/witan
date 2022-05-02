import numpy as np
import pandas as pd
from time import time
from typing import List, Dict, Any, cast

from .base import Rule, RulesetGenerator
from ..models import Classifier
from ..utils import log


class CbiRG(RulesetGenerator):
    """Generates rules from clusters produced with clustering-by-intent.

    clf: Classifier used to select residual R
    mag: Minimum accuracy gap above a random classifier for the simulated user to label a rule
    mcr: Minimum coverage of candidate features
    rn: Residual N; Number of unlabelled instances U used to construct residual R.
    mrp: Maximum residual Proportion; Maximum proportion of unlabelled instances U used to construct residual R.
    cn: Cluster N; stop constraining a cluster with terms once it has fewer than cn records in R
    mfp: Minimum false positives (i.e. instances in training set T) for a cluster to be selected.
    """

    def __init__(self, *,
                 clf: Classifier,
                 mag: float = 0.2,
                 mcr: float = 0.02,
                 rn: int = 1000,
                 mrp: float = 0.5,
                 cn: int = 25,
                 mfp: int = 50):
        self.clf = clf
        self.min_accuracy_gap = self.mag = mag
        self.min_cov_ratio = self.mcr = mcr
        self.residual_n = self.rn = rn
        self.max_residual_proportion = self.mrp = mrp
        self.cluster_n = self.cn = cn
        self.min_fp = self.mfp = mfp

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

        # Construct seed_labels_df from seed_rules
        seed_labels_df = pd.DataFrame(False, columns=classes, index=train_binary_features.index)
        seed_features = []
        for seed_rule in seed_rules:
            assert seed_rule.metadata['single_feature_rule']
            mask = train_binary_features[seed_rule.metadata['feature_name']]
            seed_labels_df[seed_rule.target_class] = seed_labels_df[seed_rule.target_class] | mask
            seed_features.append(seed_rule.metadata['feature_name'])

        log('Initialising CBI')

        init_start_time = time()

        # Training set T contains instances that have only a
        # single/consistent class label from the seed rules.
        T_mask = seed_labels_df.sum(axis=1) == 1
        T_prob_labels = seed_labels_df[T_mask].to_numpy()
        T_binary_features = train_binary_features[T_mask]
        U_binary_features = train_binary_features[~T_mask]

        # Get classifier probs for unlabelled set U.
        U_probs = self.clf.predict_probs(
            classes=classes,
            covered_train_features=T_binary_features,
            covered_train_prob_labels=T_prob_labels,
            test_features=U_binary_features,
        )

        def get_confidence(prob_row: np.array) -> float:
            max_probs = list(sorted(prob_row, reverse=True))
            return float(max_probs[0] - max_probs[1])

        # Get residual R of U classified with lowest confidence.
        U_confidences = np.apply_along_axis(get_confidence, axis=1, arr=U_probs)
        residual_n = min(self.residual_n, int(round(U_confidences.shape[0] * self.max_residual_proportion)))
        low_confidence_indexes = np.argsort(U_confidences)[:residual_n]
        R_binary_features = U_binary_features.iloc[low_confidence_indexes]

        # Pre-compute T false-positive counts, and exclude features
        # that...
        T_false_positives = T_binary_features.sum(axis=0)
        # 1. Do not appear at least min_fp times in T
        min_fp_feature_mask = T_false_positives >= self.min_fp
        # 2. Do not have minimum coverage in the full training set
        min_cov_feature_mask = train_binary_features.mean(axis=0) >= self.min_cov_ratio
        # 3. Are already used in seed rules
        non_seed_feature_mask = ~(T_binary_features.columns.isin(seed_features))
        feature_mask = min_fp_feature_mask & min_cov_feature_mask & non_seed_feature_mask
        T_false_positives = T_false_positives[feature_mask]
        selected_feature_columns = train_binary_features.columns[feature_mask]
        T_binary_features = T_binary_features[selected_feature_columns]
        R_binary_features = R_binary_features[selected_feature_columns]

        init_end_time = time()

        log('Finished initialising CBI')

        interactions = []
        for i in range(interaction_count):
            log(f'Starting CBI interaction {i}')
            interaction_start_time = time()
            # Begin constructing a new cluster C
            C_mask = pd.Series(True, index=R_binary_features.index)
            C_term_features = []
            while True:
                # Stop if we have run out of features
                if R_binary_features.shape[1] == 0:
                    break
                # Select the next query term with the highest
                # precision of differentiating C from T.
                C_true_positives = R_binary_features[C_mask].sum(axis=0)
                feature_precisions = C_true_positives / (C_true_positives + T_false_positives)
                term_feature = R_binary_features.columns[np.argmax(feature_precisions)]
                # Update C
                C_term_features.append(term_feature)
                C_mask = C_mask & R_binary_features[term_feature]
                # Reduce set of available terms
                R_binary_features = R_binary_features.drop(columns=term_feature)
                T_false_positives = T_false_positives.drop(index=term_feature)
                # Stop iterating when the cluster is sufficiently specific
                if C_mask.sum() < self.cluster_n:
                    break
            interaction_end_time = time()

            # Construct rule_series
            rule_series = None
            rule = None
            rule_useful = False
            if len(C_term_features) > 0:
                rule_mask = pd.Series(True, index=train_binary_features.index)
                for term_feature in C_term_features:
                    rule_mask = rule_mask & train_binary_features[term_feature]
                rule_covered_y = cast(pd.Series, train_y[rule_mask])
                target_class = rule_covered_y.mode().iloc[0] if rule_covered_y.shape[0] > 0 else None
                rule_accuracy = (rule_covered_y == target_class).mean()
                rule_useful = rule_accuracy >= min_accuracy
                rule = Rule(
                    target_class=target_class,
                    predicate_key='&'.join(C_term_features),
                )
                rule_series = (pd.Series(np.nan, index=train_binary_features.index)
                               .mask(rule_mask, rule.target_class)
                               .rename(rule))

            log(f'Finished interaction {i}, added {"" if rule_useful else "non-"}useful rule: {rule}')
            interactions.append({
                'rule_useful': rule_useful,
                'rule_series': rule_series,
                'wall_seconds': interaction_end_time - interaction_start_time,
            })
        return {
            'init_rules_series': [
                (pd.Series(np.nan, index=train_binary_features.index)
                 .mask(train_binary_features[rule.metadata['feature_name']], rule.target_class)
                 .rename(rule))
                for rule in seed_rules
            ],
            'init_wall_seconds': init_end_time - init_start_time,
            'interactions': interactions,
        }

    def get_rule_df(self, ruleset: Dict[str, Any], interaction_count: int) -> pd.DataFrame:
        return pd.concat([
            *ruleset['init_rules_series'],
            *[
                interaction['rule_series']
                for interaction in ruleset['interactions'][:interaction_count]
                if interaction['rule_useful']
            ],
        ], axis=1)

    def get_elapsed_wall_seconds(self, ruleset: Dict[str, Any], interaction_count: int) -> float:
        elapsed_wall_seconds = ruleset['init_wall_seconds']
        for interaction in ruleset['interactions'][:interaction_count]:
            elapsed_wall_seconds += interaction['wall_seconds']
        return elapsed_wall_seconds

    def supports_partial_interactions(self) -> bool:
        return True
