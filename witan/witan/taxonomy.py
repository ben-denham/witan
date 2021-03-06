"""Classes for taxonomies of LF conditions generated by Witan."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Set, Callable, Sequence, cast
from copy import copy

NOT_TOKEN = '^'
OR_TOKEN = '/'
AND_TOKEN = '&'


class Condition(ABC):
    """Base class for LF conditions."""

    @abstractmethod
    def get_mask(self, bow_df: pd.DataFrame) -> np.ndarray:
        """Return a Boolean array indicating whether this condition is true
        for each row in the given bag-of-words DataFrame."""
        pass

    @abstractmethod
    def serialize(self) -> str:
        """Return a string representation of this condition."""
        pass

    @abstractmethod
    def get_all_columns(self) -> Set[str]:
        """Return a set of all columns used directly and indirectly by this condition."""
        pass

    def __repr__(self) -> str:
        return self.serialize()

    def __str__(self) -> str:
        return self.serialize()

    def __lt__(self, other: 'Condition') -> bool:
        return self.serialize() < other.serialize()


class UnaryCondition(Condition):
    """Base class for LF conditions whose serializations do not need to be
    wrapped in parentheses."""
    pass


@dataclass(eq=True, frozen=True)
class NullCondition(UnaryCondition):
    """Condition that returns true for all inputs."""

    def get_mask(self, bow_df: pd.DataFrame) -> np.ndarray:
        return np.ones(bow_df.shape[0]).astype(bool)

    def serialize(self) -> str:
        return ''

    def get_all_columns(self) -> Set[str]:
        return set()


@dataclass(eq=True, frozen=True)
class OrCondition(Condition):
    """Condition that performs a logical OR of other conditions."""
    or_conditions: Tuple[Condition]

    def get_mask(self, bow_df: pd.DataFrame) -> np.ndarray:
        if not self.or_conditions:
            return np.ones(bow_df.shape[0]).astype(bool)
        condition_mask = np.zeros(bow_df.shape[0]).astype(bool)
        for condition in self.or_conditions:
            condition_mask = condition_mask | condition.get_mask(bow_df)
        return condition_mask

    def serialize(self) -> str:
        serialized_conditions = []
        for condition in self.or_conditions:
            serialized_condition = condition.serialize()
            if not isinstance(condition, UnaryCondition):
                serialized_condition = f'({serialized_condition})'
            serialized_conditions.append(serialized_condition)
        return OR_TOKEN.join(serialized_conditions)

    def get_all_columns(self) -> Set[str]:
        return cast(Set[str], set()).union(*[
            condition.get_all_columns() for condition in self.or_conditions
        ])


@dataclass(eq=True, frozen=True)
class AndCondition(Condition):
    """Condition that performs a logical AND of other conditions."""
    and_conditions: Tuple[Condition]

    def get_mask(self, bow_df: pd.DataFrame) -> np.ndarray:
        if not self.and_conditions:
            return np.ones(bow_df.shape[0]).astype(bool)
        condition_mask = np.ones(bow_df.shape[0]).astype(bool)
        for condition in self.and_conditions:
            condition_mask = condition_mask & condition.get_mask(bow_df)
        return condition_mask

    def serialize(self) -> str:
        serialized_conditions = []
        for condition in self.and_conditions:
            serialized_condition = condition.serialize()
            if not isinstance(condition, UnaryCondition):
                serialized_condition = f'({serialized_condition})'
            serialized_conditions.append(serialized_condition)
        return AND_TOKEN.join(serialized_conditions)

    def get_all_columns(self) -> Set[str]:
        return cast(Set[str], set()).union(*[
            condition.get_all_columns() for condition in self.and_conditions
        ])


@dataclass(eq=True, frozen=True)
class NotCondition(UnaryCondition):
    """Condition that inverts the result of another condition."""
    notted_condition: Condition

    def get_mask(self, bow_df: pd.DataFrame) -> np.ndarray:
        return ~(self.notted_condition.get_mask(bow_df))

    def serialize(self) -> str:
        serialized_condition = self.notted_condition.serialize()
        if not isinstance(self.notted_condition, UnaryCondition):
            serialized_condition = f'({serialized_condition})'
        return f'{NOT_TOKEN}{serialized_condition}'

    def get_all_columns(self) -> Set[str]:
        return self.notted_condition.get_all_columns()


@dataclass(eq=True, frozen=True)
class FeatureCondition(UnaryCondition):
    """Condition based on a Boolean feature."""
    feature_name: str

    def __post_init__(self):
        assert NOT_TOKEN not in self.feature_name
        assert OR_TOKEN not in self.feature_name
        assert AND_TOKEN not in self.feature_name

    def get_mask(self, bow_df: pd.DataFrame) -> np.ndarray:
        try:
            return bow_df[self.feature_name].to_numpy().astype(bool)
        except KeyError:
            # If feature_name is not in bow_df, then the condition
            # matches no instances.
            return np.zeros(bow_df.shape[0]).astype(bool)

    def serialize(self) -> str:
        return self.feature_name

    def get_all_columns(self) -> Set[str]:
        return set([self.feature_name])


@dataclass
class TaxonomyNode:
    """Representation of a node in a taxonomy tree of LF conditions."""
    parent: Optional['TaxonomyNode'] = None
    condition: Condition = field(default_factory=NullCondition)
    children: List['TaxonomyNode'] = field(default_factory=list)
    target_class: Optional[str] = None
    useful: Optional[bool] = None

    @property
    def short_serialized(self) -> str:
        """Serialization of just this node's condition without the parent condition."""
        return self.condition.serialize()

    @property
    def serialized(self) -> str:
        """Serialization of this taxonomy node."""
        result = self.condition.serialize()
        if not isinstance(self.condition, UnaryCondition):
            result = f'({result})'
        if self.parent:
            parent_serialized = self.parent.serialized
            if len(parent_serialized) > 0:
                result = f'{parent_serialized}&{result}'
        return result

    @property
    def short_name(self) -> str:
        """User-friendly representation of this node."""
        if isinstance(self.condition, NullCondition):
            return 'ALL'
        return self.short_serialized

    def get_all_columns(self) -> Set[str]:
        """Returns all columns used in the conditions of this node and its parent."""
        parent_columns = set() if self.parent is None else self.parent.get_all_columns()
        return parent_columns.union(self.condition.get_all_columns())

    def get_parent_mask(self, bow_df: pd.DataFrame) -> np.ndarray:
        """Return the mask for this node's parent."""
        return (
            self.parent.get_mask(bow_df) if self.parent is not None
            else np.ones(bow_df.shape[0]).astype(bool)
        )

    def get_condition_mask(self, bow_df: pd.DataFrame) -> np.ndarray:
        """Return the mask for this node's condition, without its parent's condition."""
        return self.condition.get_mask(bow_df)

    def get_mask(self, bow_df: pd.DataFrame) -> np.ndarray:
        """Return a Boolean array indicating whether this node matches
        each row in the given bag-of-words DataFrame."""
        return self.get_parent_mask(bow_df) & self.get_condition_mask(bow_df)

    def apply_recursive(self, func: Callable[['TaxonomyNode'], None]):
        """Apply the given func to this node and all its children."""

        def apply_func(node):
            func(node)
            for child in node.children:
                apply_func(child)

        apply_func(self)


def prune_taxonomy(taxonomy: TaxonomyNode, *,
                   keep_nodes: Sequence[TaxonomyNode],
                   parent: Optional[TaxonomyNode] = None):
    """Returns a new taxonomy that removes any nodes not in keep_nodes
    from the given taxonomy."""
    new_taxonomy = copy(taxonomy)
    if parent is not None:
        new_taxonomy.parent = parent
    new_taxonomy.children = [
        prune_taxonomy(child, keep_nodes=keep_nodes, parent=new_taxonomy)
        for child in new_taxonomy.children
        if child in keep_nodes
    ]
    return new_taxonomy


def get_columns_in_taxonomy(taxonomy: TaxonomyNode, all_columns: Set[str]) -> Set[str]:
    """Return the set of all columns used in the given taxonomy."""
    columns: Set[str] = set()

    def search_node(node: TaxonomyNode):
        nonlocal columns
        columns = columns.union(node.condition.get_all_columns())

    taxonomy.apply_recursive(search_node)
    # Remove any columns that don't exist in the set of all columns.
    return columns.intersection(all_columns)
