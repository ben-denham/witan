from abc import ABC, abstractmethod
from time import time
import numpy as np
import pandas as pd
from typing import List, Dict, Any

from ..utils import KeyHashable
from ..utils.rules import Rule


class RulesetGenerator(ABC, KeyHashable):
    """Base class for rule/LF set generators."""

    @abstractmethod
    def generate_ruleset(self,
                         classes: np.array,
                         train_binary_features: pd.DataFrame,
                         train_y: pd.Series,
                         seed_rules: List[Rule],
                         rngseed: int,
                         interaction_count: int) -> Any:
        """Generates a ruleset for given training data, seed rules, and
        interaction count."""
        pass

    @abstractmethod
    def get_rule_df(self, ruleset: Any, interaction_count: int) -> pd.DataFrame:
        """Return a rule_df from the given ruleset for the given interaction_count."""
        pass

    @abstractmethod
    def get_elapsed_wall_seconds(self, ruleset: Any, interaction_count: int) -> float:
        """Return runtime from the given ruleset for the given interaction_count."""
        pass

    def supports_partial_interactions(self) -> bool:
        """Returns True if a ruleset generated by this class can also be used
        to retrieve rulesets for lower interaction counts."""
        return False

    def get_extras(self, ruleset: Any, interaction_count: int) -> Dict[str, Any]:
        """Return extra results from the given ruleset for the given interaction_count."""
        return {}


class TrueRG(RulesetGenerator):
    """Generates a single rule that reflects the full training dataset."""

    def generate_ruleset(self,
                         classes: np.array,
                         train_binary_features: pd.DataFrame,
                         train_y: pd.Series,
                         seed_rules: List[Rule],
                         rngseed: int,
                         interaction_count: int) -> Dict[str, Any]:
        start_time = time()
        rule_df = pd.DataFrame({
            Rule(target_class, 'truth'): train_y.mask(train_y != target_class, np.nan)
            for target_class in classes
        })
        end_time = time()
        return {
            'rule_df': rule_df,
            'elapsed_wall_seconds': end_time - start_time,
        }

    def get_rule_df(self, ruleset: Dict[str, Any], interaction_count: int) -> pd.DataFrame:
        return ruleset['rule_df']

    def get_elapsed_wall_seconds(self, ruleset: Dict[str, Any], interaction_count: int) -> float:
        return ruleset['elapsed_wall_seconds']

    def supports_partial_interactions(self) -> bool:
        return True
