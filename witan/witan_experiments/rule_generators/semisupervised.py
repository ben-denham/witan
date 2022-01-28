import numpy as np
import pandas as pd
from time import time
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.semi_supervised import LabelSpreading
from sklearn.neighbors import NearestNeighbors
from typing import List, Dict, Any, Optional

from .base import Rule, RulesetGenerator
from ..utils import log

# The special unknown value expected by LabelSpreading.
UNKNOWN_LABEL = -1


class SparseKnnLabelSpreading(LabelSpreading):
    """Extend sklearn.semi_supervised.LabelSpreading to support sparse
    matrices."""
    nn_fit: Optional[NearestNeighbors]

    def _get_kernel(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        sparse_X = csr_matrix(X)
        if self.nn_fit is None:
            self.nn_fit = NearestNeighbors(n_neighbors=self.n_neighbors,
                                           n_jobs=self.n_jobs).fit(sparse_X)
        if y is None:
            return self.nn_fit.kneighbors_graph(self.nn_fit._fit_X,
                                                self.n_neighbors,
                                                mode='connectivity')
        else:
            return self.nn_fit.kneighbors(y, return_distance=False)


class SemiSupervisedRG(RulesetGenerator):
    """Generates a single rule that assigns instance labels based on label
    spreading from a random sample of instances."""

    def generate_ruleset(self,
                         classes: np.array,
                         train_binary_features: pd.DataFrame,
                         train_y: pd.Series,
                         seed_rules: List[Rule],
                         rngseed: int,
                         interaction_count: int) -> Dict[str, Any]:
        X = train_binary_features.to_numpy()
        y = train_y.to_numpy()

        le = LabelEncoder().fit(y)
        y = le.fit_transform(y)
        unlbld_mask = np.full(y.shape, fill_value=True)
        unlbld_mask[:interaction_count] = False
        np.random.RandomState(rngseed).shuffle(unlbld_mask)
        y[unlbld_mask] = UNKNOWN_LABEL

        start_time = time()

        model = SparseKnnLabelSpreading(kernel='knn')
        log('Fitting LabelSpreading')
        model.fit(X, y)
        log('Applying LabelSpreading')
        prob_y = model.label_distributions_

        end_time = time()

        # Some instances may be assigned a prob of 0 for all classes,
        # so we will keep those instances labelled as np.nan.
        pred_y = np.full(y.shape, fill_value=np.nan, dtype=np.object)
        pos_prob_mask = prob_y.sum(axis=1) > 0
        pred_y[pos_prob_mask] = le.inverse_transform(
            model.classes_[np.argmax(prob_y[pos_prob_mask], axis=1)])

        rule_df = pd.DataFrame(
            {Rule(target_class=None, predicate_key='semi-supervised'): pred_y},
            index=train_binary_features.index,
        )

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
