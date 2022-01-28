from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Dict, List

from .utils import KeyHashable
from .utils.rules import Rule, make_single_feature_rule


class RuleSeeder(ABC, KeyHashable):
    """Base class for seed rule/LF generators."""

    @abstractmethod
    def get_seed_rules(self,
                       classes: np.array,
                       train_binary_features: pd.DataFrame,
                       train_y: pd.Series,
                       rngseed: int) -> List[Rule]:
        """Generates a set of seed rules/LFs."""
        pass


class BlankRS(RuleSeeder):
    """Generates no seed rules"""

    def get_seed_rules(self,
                       classes: np.array,
                       train_binary_features: pd.DataFrame,
                       train_y: pd.Series,
                       rngseed: int) -> List[Rule]:
        return []


class AccRS(RuleSeeder):
    """Generates seed rules for random features within a range of
    acceptable accuracies:

    min_cov_ratio: The minimum coverage for each seed rule.
    min_accuracy_gap: The minimum accuracy gap above a random classifier of seed rules.
    max_accuracy_gap: The maximum accuracy gap above a random classifier of seed rules.
    rpc: The number of seed rules to generate for each class.
    """

    def __init__(self,
                 min_cov_ratio: float = 0.02,
                 min_acc_gap: float = 0.2,
                 max_acc_gap: float = 0.35,
                 rpc: int = 2):
        self.min_cov_ratio = min_cov_ratio
        self.min_accuracy_gap = self.min_acc_gap = min_acc_gap
        self.max_accuracy_gap = self.max_acc_gap = max_acc_gap
        self.rules_per_class = self.rpc = rpc

    def check_accuracy(self, accuracy: float, *, class_count: int) -> bool:
        """Check whether the given accuracy is within the acceptable range."""
        chance_accuracy = 1 / class_count
        if (self.min_accuracy_gap is not None
                and accuracy < (chance_accuracy + self.min_accuracy_gap)):
            return False
        if (self.max_accuracy_gap is not None
                and accuracy > (chance_accuracy + self.max_accuracy_gap)):
            return False
        return True

    def get_potential_class_rules(self,
                                  classes: np.array,
                                  train_binary_features: pd.DataFrame,
                                  train_y: pd.Series) -> Dict[str, List[Rule]]:
        """Return all potential rules for each class that meet requirements."""
        column_mask = train_binary_features.mean() >= self.min_cov_ratio
        train_binary_features = train_binary_features.loc[:, column_mask]

        potential_class_rules: Dict[str, List[Rule]] = {
            target_class: [] for target_class in classes
        }
        for feature_name in train_binary_features.columns:
            covered_mask = train_binary_features[feature_name]
            covered_y = train_y[covered_mask]
            class_proportions = covered_y.value_counts(normalize=True, sort=True)
            most_frequent_class = class_proportions.index[0]
            if self.check_accuracy(class_proportions[most_frequent_class],
                                   class_count=classes.shape[0]):
                rule = make_single_feature_rule(most_frequent_class, feature_name)
                potential_class_rules[most_frequent_class].append(rule)
        return potential_class_rules

    def get_seed_rules(self,
                       classes: np.array,
                       train_binary_features: pd.DataFrame,
                       train_y: pd.Series,
                       rngseed: int) -> List[Rule]:
        potential_class_rules = self.get_potential_class_rules(
            classes=classes,
            train_binary_features=train_binary_features,
            train_y=train_y,
        )

        rng = np.random.RandomState(rngseed)
        seed_rules = []
        for target_class in classes:
            if len(potential_class_rules[target_class]) < self.rules_per_class:
                raise ValueError(f'Not enough potential seed rules for class: {target_class}')
            seed_rules += rng.choice(potential_class_rules[target_class],
                                     self.rules_per_class,
                                     replace=False).tolist()
        return seed_rules
