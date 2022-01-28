import numpy as np
import pandas as pd
from time import time
from sklearn.model_selection import train_test_split
from typing import List, Dict, Any

from ...utils import log
from ..base import Rule, RulesetGenerator
from .reef.heuristic_generator import HeuristicGenerator, ABSTAIN_LABEL


class SnubaRG(RulesetGenerator):
    """Generates rules/LFs using Snuba based on a random sample of
    instances equal to the number of allowed user interactions.

    mcr: Minimum coverage of features included in the set of possible rules
    init_rules: How many rules to keep on the first iteration.
    max_rules: Maximum number of rules to generate.
    card: Allowed cardinality of decision tree primitives.

    Snuba/Reef source-code from:
    https://github.com/HazyResearch/reef/tree/bc7c1ccaf40ea7bf8f791035db551595440399e3
    used under included license.
    """

    def __init__(self,
                 mcr: float = 0.02,
                 init_rules: int = 3,
                 max_rules: int = 20,
                 card: int = 1):
        self.min_cov_ratio = self.mcr = mcr
        self.init_rules = init_rules
        self.max_rules = max_rules
        self.card = card

    def generate_ruleset(self,
                         classes: np.array,
                         train_binary_features: pd.DataFrame,
                         train_y: pd.Series,
                         seed_rules: List[Rule],
                         rngseed: int,
                         interaction_count: int) -> Dict[str, Any]:
        full_binary_features = train_binary_features.loc[:, train_binary_features.mean() >= self.min_cov_ratio]
        unlbld_binary_features, lbld_binary_features, _, lbld_y = train_test_split(
            full_binary_features.to_numpy(), train_y.to_numpy(),
            test_size=interaction_count, random_state=rngseed)

        start_time = time()

        hg = HeuristicGenerator(unlbld_binary_features, lbld_binary_features, lbld_y)

        feedback_idx = None
        previous_iteration = 0
        for iteration in range(self.init_rules, self.max_rules + 1):
            keep_rules = self.init_rules if (iteration == self.init_rules) else 1
            hg.run_synthesizer(
                max_cardinality=self.card,
                idx=feedback_idx,
                keep=keep_rules,
                model='dt',
            )
            hg.run_verifier()
            hg.find_feedback()

            new_rules_desc = ';'.join([
                ','.join(full_binary_features.columns[list(feat_idxs)])
                for feat_idxs in hg.feat_combos[previous_iteration:iteration]
            ])
            log(f'Completed iteration {iteration}: {new_rules_desc}')

            previous_iteration = iteration
            feedback_idx = hg.feedback_idx
            if feedback_idx == []:
                break

        end_time = time()

        beta_opt = hg.syn.find_optimal_beta(hg.hf, hg.val_primitive_matrix,
                                            hg.feat_combos, hg.val_ground)
        L = hg.apply_heuristics(hg.hf, full_binary_features.to_numpy(), hg.feat_combos, beta_opt)
        rules = [
            Rule(
                target_class=None,
                predicate_key=f'{i}({full_binary_features.columns[feat_idxs]})',
                metadata={'hfs': hfs},
            )
            for i, (hfs, feat_idxs) in enumerate(zip(hg.hf, hg.feat_combos))
        ]
        rule_df = pd.DataFrame(L, columns=rules, index=train_binary_features.index).replace({
            # Replace ABSTAIN with nan
            ABSTAIN_LABEL: np.nan,
            # Replace class values with labels.
            **{class_val: target_class
               for class_val, target_class in zip(hg.classes, hg.class_labels)}
        })

        return {
            'rule_df': rule_df,
            'elapsed_wall_seconds': end_time - start_time,
        }

    def get_rule_df(self, ruleset: Dict[str, Any], interaction_count: int) -> pd.DataFrame:
        return ruleset['rule_df']

    def get_elapsed_wall_seconds(self, ruleset: Dict[str, Any], interaction_count: int) -> float:
        return ruleset['elapsed_wall_seconds']

    def supports_partial_interactions(self) -> bool:
        return False
