from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from snorkel.labeling.model.baselines import MajorityLabelVoter
from snorkel.labeling.model import LabelModel
import torch

from .utils import KeyHashable

SNORKEL_ABSTAIN = -1


def L_from_rule_df(classes: np.array, rule_df: pd.DataFrame) -> np.array:
    """Convert rule_df to L array required by Snorkel, with class indexes
    in place of class labels and SNORKEL_ABSTAIN in place of None."""
    # Convert from sparse to dense if not already dense.
    try:
        rule_df = rule_df.sparse.to_dense()
    except AttributeError as ex:
        if "Can only use the '.sparse' accessor with Sparse data." in str(ex):
            pass
        else:
            raise ex

    rule_df = rule_df.replace({target_class: i for i, target_class in enumerate(classes)})
    rule_df = rule_df.fillna(SNORKEL_ABSTAIN)
    return rule_df.to_numpy().astype(int)


class Labeller(ABC, KeyHashable):
    """Base class for Labellers that convert rule_dfs into probabilistic labels."""

    @abstractmethod
    def get_probabilistic_labels(self,
                                 classes: np.array,
                                 rngseed: int,
                                 rule_df: pd.DataFrame) -> np.ndarray:
        """Return probabalistic labels DataFrame for given rule_df."""
        pass


class MVLblr(Labeller):
    """Perform labelling by majority vote."""

    def get_probabilistic_labels(self,
                                 classes: np.array,
                                 rngseed: int,
                                 rule_df: pd.DataFrame) -> np.ndarray:
        L = L_from_rule_df(classes, rule_df)
        probs = MajorityLabelVoter(cardinality=len(classes)).predict_proba(L)
        assert not np.any(np.isnan(probs))
        return probs


class SnorkelLblr(Labeller):
    """Perform labelling by Snorkel's generative model. If mv_cb=True,
    then majority vote is used to determine the assumed class distribution,
    otherwise perfectly balanced classes are assumed."""

    def __init__(self, mv_cb: bool = False):
        self.mv_class_balance = self.mv_cb = mv_cb

    def get_probabilistic_labels(self,
                                 classes: np.array,
                                 rngseed: int,
                                 rule_df: pd.DataFrame) -> np.ndarray:
        # Snorkel requires at least 3 rules, so fallback to majority vote.
        if rule_df.shape[1] < 3:
            return MVLblr().get_probabilistic_labels(classes, rngseed, rule_df)

        L = L_from_rule_df(classes, rule_df)
        cardinality = len(classes)

        if self.mv_class_balance:
            mvs = MajorityLabelVoter(cardinality=cardinality).predict_proba(L)
            class_balance = np.mean(mvs, axis=0)
            class_balance = class_balance / class_balance.sum()
        else:
            class_balance = np.ones(cardinality) / cardinality

        device = 'cpu'
        if 'cuda' in device:
            torch.cuda.empty_cache()
            label_model = LabelModel(cardinality=cardinality, verbose=True, device=device)
            label_model.fit(L, class_balance=class_balance, seed=rngseed)
            torch.cuda.empty_cache()
        else:
            label_model = LabelModel(cardinality=cardinality, verbose=True)
            label_model.fit(L, class_balance=class_balance, seed=rngseed)

        probs = label_model.predict_proba(L)
        assert not np.any(np.isnan(probs))
        return probs
