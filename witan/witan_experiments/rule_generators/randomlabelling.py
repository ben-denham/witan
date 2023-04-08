import numpy as np
import pandas as pd
from typing import List, Dict, Any

from .base import Rule, RulesetGenerator


class RandomLabellingRG(RulesetGenerator):
    """Generates a single rule that labels a random set of instances."""

    def generate_ruleset(self,
                         classes: np.array,
                         train_binary_features: pd.DataFrame,
                         train_y: pd.Series,
                         seed_rules: List[Rule],
                         rngseed: int,
                         interaction_count: int) -> Dict[str, Any]:
        rng = np.random.RandomState(rngseed)
        y_array = train_y.to_numpy()

        iterations = []
        selected_idxs = rng.choice(np.arange(y_array.shape[0]), size=interaction_count, replace=False)
        for selected_idx in selected_idxs:
            iterations.append({
                'selected_index': selected_idx,
                'instance_target_class': y_array[selected_idx],
                'wall_seconds': 0,
            })

        return {
            'train_index': train_binary_features.index,
            'iterations': iterations,
        }

    def get_rule_df(self, ruleset: Dict[str, Any], interaction_count: int) -> pd.DataFrame:
        y = np.full(ruleset['train_index'].shape[0], fill_value=np.nan).astype(np.object)

        for iteration in ruleset['iterations'][:interaction_count]:
            y[iteration['selected_index']] = iteration['instance_target_class']

        return pd.DataFrame({
            Rule(target_class=None, predicate_key='random-sampling'): y,
        }, index=ruleset['train_index'])

    def get_elapsed_wall_seconds(self, ruleset: Dict[str, Any], interaction_count: int) -> float:
        elapsed_wall_seconds = 0
        for iteration in ruleset['iterations'][:interaction_count]:
            elapsed_wall_seconds += iteration['wall_seconds']
        return elapsed_wall_seconds

    def supports_partial_interactions(self) -> bool:
        return True
