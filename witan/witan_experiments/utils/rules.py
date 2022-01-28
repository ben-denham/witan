from dataclasses import dataclass, replace, field
from typing import Optional, Any, Dict


@dataclass(eq=True, frozen=True)
class Rule:
    """Common representation of rules returned by all rule_seeders and
    rule_generators."""

    target_class: Optional[str]
    predicate_key: str
    # Metadata will not affect equality of Rules.
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return f'{self.target_class}|{self.predicate_key}'

    def __str__(self) -> str:
        return self.__repr__()

    def __hash__(self) -> int:
        return hash(self.__repr__())

    def __eq__(self, other: Any) -> bool:
        return self.__hash__() == other.__hash__()

    def update(self, **changes: Dict[str, Any]) -> 'Rule':
        return replace(self, **changes)


def make_single_feature_rule(target_class: str, feature_name: str) -> Rule:
    """Return a rule representing a condition on a single feature and
    assigning the given target_class."""
    return Rule(
        target_class=target_class,
        predicate_key=feature_name,
        metadata={
            'single_feature_rule': True,
            'feature_name': feature_name,
        }
    )
