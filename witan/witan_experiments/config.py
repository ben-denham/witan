from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from itertools import product, groupby
from typing import Any, Sequence, List, Set, Dict
import os

from .models import Classifier
from .rule_seeders import RuleSeeder
from .rule_generators import RulesetGenerator
from .labellers import Labeller


class Config(ABC):

    @abstractmethod
    def cache_key(self) -> str:
        """Return a unique key for a given config that can be used as a
        directory path."""
        pass

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass(eq=True, frozen=True)
class PrepConfig(Config):
    """Config for dataset preparation."""
    dataset_name: str

    def cache_key(self) -> str:
        return os.path.join(
            self.dataset_name,
        )


@dataclass(eq=True, frozen=True)
class RulesConfig(Config):
    """Config for generating rules/LFs."""
    prep_config: PrepConfig
    rule_seeder: RuleSeeder
    ruleset_generator: RulesetGenerator
    rngseed: int
    interaction_count: int

    def cache_key_without_interactions(self) -> str:
        return os.path.join(
            self.prep_config.cache_key(),
            f'{self.rule_seeder.key()}-{self.ruleset_generator.key()}-rs{self.rngseed}',
        )

    def cache_key(self) -> str:
        return f'{self.cache_key_without_interactions()}-ic{self.interaction_count}'

    def partial_interactions_key(self) -> str:
        """Return a unique cache key that accounts for the fact that cached
        results for methods supporting partial interactions can have
        be re-used for different interaction counts."""
        if self.ruleset_generator.supports_partial_interactions():
            return self.cache_key_without_interactions()
        else:
            return self.cache_key()


@dataclass(eq=True, frozen=True)
class LabelsConfig(Config):
    """Config for generating probabilistic labels from rules."""
    rules_config: RulesConfig
    labeller: Labeller

    def cache_key(self) -> str:
        return os.path.join(
            self.rules_config.cache_key(),
            self.labeller.key(),
        )


@dataclass(eq=True, frozen=True)
class ExperimentConfig(Config):
    """Config for a complete experiment, including classification."""
    dataset_name: str
    rule_seeder: RuleSeeder
    rngseed: int
    ruleset_generator: RulesetGenerator
    interaction_count: int
    labeller: Labeller
    classifier: Classifier

    @property
    def prep_config(self) -> PrepConfig:
        return PrepConfig(
            dataset_name=self.dataset_name,
        )

    @property
    def rules_config(self) -> RulesConfig:
        return RulesConfig(
            prep_config=self.prep_config,
            rule_seeder=self.rule_seeder,
            ruleset_generator=self.ruleset_generator,
            rngseed=self.rngseed,
            interaction_count=self.interaction_count,
        )

    @property
    def labels_config(self) -> LabelsConfig:
        return LabelsConfig(
            rules_config=self.rules_config,
            labeller=self.labeller,
        )

    def cache_key(self) -> str:
        return os.path.join(
            self.labels_config.cache_key(),
            self.classifier.key(),
        )


def prepare_experiment_configs(**kwargs: List[Any]) -> Sequence[ExperimentConfig]:
    """Prepare ExperimentConfigs for the cartesian product of provided
    multi-valued config parameters."""
    keys, value_lists = kwargs.keys(), kwargs.values()
    return [
        ExperimentConfig(**dict(zip(keys, values)))
        for values in product(*value_lists)
    ]


def condense_rules_configs(rules_configs: Set[RulesConfig]) -> Dict[RulesConfig, List[RulesConfig]]:
    """Return a map of the minimal rules_configs that must be executed to
    the rules_configs with different interaction counts they support."""

    def group_key(rules_config: RulesConfig) -> str:
        return rules_config.partial_interactions_key()

    rules_config_groups = {
        key: list(group) for key, group in
        groupby(sorted(rules_configs, key=group_key), group_key)
    }
    rules_config_map = {}
    for rules_config_group in rules_config_groups.values():
        max_interaction_rules_config = max(rules_config_group,
                                           key=lambda rules_config: rules_config.interaction_count)
        rules_config_map[max_interaction_rules_config] = rules_config_group
    return rules_config_map
