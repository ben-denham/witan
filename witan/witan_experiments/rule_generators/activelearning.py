import numpy as np
import pandas as pd
from time import time
from typing import List, Dict, Set, Any

from .base import Rule, RulesetGenerator
from ..models import Classifier
from ..utils import log


class ActiveLearningRG(RulesetGenerator):
    """Generates a single rule that labels a set of instances iteratively
    constructed with uncertainty sampling.

    clf: A classifier object to perform uncertainty sampling with.
    init_count: The number of instances to randomly sample from each
                class initially. If zero, instances are sampled randomly
                until at least 2 classes are identified to begin
                uncertainty sampling with.
    """

    def __init__(self, *, clf: Classifier, init_count: int = 0):
        self.clf = clf
        self.init_count = init_count

    def generate_ruleset(self,
                         classes: np.array,
                         train_binary_features: pd.DataFrame,
                         train_y: pd.Series,
                         seed_rules: List[Rule],
                         rngseed: int,
                         interaction_count: int) -> Dict[str, Any]:
        rng = np.random.RandomState(rngseed)
        class_to_index = {target_class: i for i, target_class in enumerate(classes)}
        y_array = train_y.to_numpy()

        # Initialise the state arrays that we update as we query instances.
        lbld_mask = np.full(y_array.shape, fill_value=False)
        train_prob_labels = np.zeros((y_array.shape[0], classes.shape[0]))

        def learn_from_index(idx):
            """Update state arrays by querying the given instance idx."""
            lbld_mask[idx] = True
            target_class = y_array[idx]
            class_idx = class_to_index[target_class]
            train_prob_labels[idx, class_idx] = 1

        iterations = []
        init_idx_to_class = {}
        if self.init_count > 0:
            # Initialise training set with init_count instances from
            # each class.
            for target_class in classes:
                class_idxs = np.nonzero(y_array == target_class)[0]
                selected_idxs = rng.choice(class_idxs, self.init_count)
                for selected_idx in selected_idxs:
                    init_idx_to_class[selected_idx] = y_array[selected_idx]
                    learn_from_index(selected_idx)
        else:
            # Randomly sample until we have 2 classes (needed to begin
            # uncertainty sampling).
            discovered_classes: Set[str] = set()
            while len(discovered_classes) < 2:
                start_time = time()
                selected_idx = rng.randint(0, y_array.shape[0])
                end_time = time()
                discovered_classes.add(y_array[selected_idx])
                learn_from_index(selected_idx)
                iterations.append({
                    'selected_index': selected_idx,
                    'instance_target_class': y_array[selected_idx],
                    'wall_seconds': end_time - start_time,
                })

        for i in range(len(iterations), interaction_count):
            log(f'Active learning iteration: {i + 1}')
            start_time = time()
            probs = self.clf.predict_probs(
                classes=classes,
                covered_train_features=train_binary_features[lbld_mask],
                covered_train_prob_labels=train_prob_labels[lbld_mask],
                test_features=train_binary_features,
            )
            end_time = time()
            # Select row with minimum confidence to learn from.
            confidences = np.max(probs, axis=1)
            active_selection_idx = np.argmin(confidences + lbld_mask)
            learn_from_index(active_selection_idx)
            iterations.append({
                'selected_index': active_selection_idx,
                'instance_target_class': y_array[active_selection_idx],
                'wall_seconds': end_time - start_time,
            })

        return {
            'init_idx_to_class': init_idx_to_class,
            'train_index': train_binary_features.index,
            'iterations': iterations,
        }

    def get_rule_df(self, ruleset: Dict[str, Any], interaction_count: int) -> pd.DataFrame:
        y = np.full(ruleset['train_index'].shape[0], fill_value=np.nan).astype(np.object)

        for idx, target_class in ruleset['init_idx_to_class'].items():
            y[idx] = target_class

        for iteration in ruleset['iterations'][:interaction_count]:
            y[iteration['selected_index']] = iteration['instance_target_class']

        return pd.DataFrame({
            Rule(target_class=None, predicate_key='active-learning'): y,
        }, index=ruleset['train_index'])

    def get_elapsed_wall_seconds(self, ruleset: Dict[str, Any], interaction_count: int) -> float:
        elapsed_wall_seconds = 0
        for iteration in ruleset['iterations'][:interaction_count]:
            elapsed_wall_seconds += iteration['wall_seconds']
        return elapsed_wall_seconds

    def supports_partial_interactions(self) -> bool:
        return True
