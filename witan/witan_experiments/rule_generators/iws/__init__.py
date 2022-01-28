from abc import abstractmethod
from itertools import product
from time import time
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from typing import List, Dict, Any, Union, cast

from .iws import InteractiveWeakSupervision
from ..base import Rule, RulesetGenerator
from ...utils import log
from ...utils.rules import make_single_feature_rule


class IWSRG(RulesetGenerator):
    """Generates rules/LFs using IWS with an iteration for each allowed
    user interaction.

    acq: Selects either the 'AS' or 'LSE' acquisition function. LSE
         acquisition has the behaviour of 'LSE-AC'
    mag: Minimum accuracy gap above a random classifier for the simulated
         user to approve a rule
    mcr: Minimum coverage of features included in the LF space
    """

    def __init__(self,
                 acq: str = 'AS',
                 mag: float = 0.2,
                 mcr: float = 0.02):
        self.acquisition = self.acq = acq
        self.min_accuracy_gap = self.mag = mag
        self.min_cov_ratio = self.mcr = mcr

    def get_min_accuracy(self, classes: np.array) -> float:
        """Return the minimum accuracy for a simulated user to label a
        rule."""
        if self.min_accuracy_gap is None:
            return 0
        chance_accuracy = 1 / classes.shape[0]
        return chance_accuracy + self.min_accuracy_gap

    def get_oracle_func(self,
                        lfs: pd.DataFrame,
                        train_y: pd.Series,
                        min_accuracy: float) -> np.array:
        """Returns a function that acts as a simulated user to approve/reject
        one of the given LFs at a certain index based on the accuracy
        against train_y."""
        def oracle_func(idx: int) -> bool:
            rule = lfs.columns[idx]
            rule_covered_y = train_y[lfs[rule]]
            rule_mode_class = rule_covered_y.mode().iloc[0]
            if rule_mode_class != rule.target_class:
                return False
            rule_accuracy = (rule_covered_y == rule.target_class).mean()
            return rule_accuracy >= min_accuracy
        return oracle_func

    def get_lf_features(self, classes: np.array, lfs: pd.DataFrame) -> np.array:
        """Return SVD-projected representation of LF features for IWS model."""
        binary_lf_feature_rows_array = self.get_raw_lf_feature_rows(classes, lfs)
        svd = TruncatedSVD(n_components=150, n_iter=20, random_state=42)
        return svd.fit_transform(binary_lf_feature_rows_array).astype(np.float32)

    def lfs_to_rule_df(self, lfs: pd.DataFrame) -> pd.DataFrame:
        """Convert IWS LFs to rule_df."""
        def rule_column_builder(column: pd.Series) -> pd.Series:
            rule_column = pd.Series(np.repeat(np.nan, column.shape[0]),
                                    dtype=pd.SparseDtype(np.dtype('object'), fill_value=np.nan),
                                    index=column.index)
            rule_column = rule_column.mask(column, cast(Rule, column.name).target_class)
            return rule_column
        rules = lfs.apply(rule_column_builder, axis=0)
        return cast(pd.DataFrame, rules)

    def get_possible_rules(self,
                           classes: np.array,
                           train_binary_features: pd.DataFrame,
                           train_y: pd.Series) -> List[Rule]:
        """Return rules for an LF space of simple conditions of the given
        binary features returning any given class."""
        return [
            make_single_feature_rule(target_class, feature_name)
            for target_class, feature_name
            in product(classes, train_binary_features.columns)
        ]

    def get_lfs_for_rules(self,
                          rules: List[Rule],
                          train_binary_features: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame of IWS LFs for the given rules."""
        lf_columns = {}
        for rule in rules:
            feature_name = rule.metadata['feature_name']
            lf_columns[rule] = train_binary_features[feature_name]
        return pd.DataFrame(lf_columns)

    def get_iteration_lf_idxs(self, *,
                              classes: np.ndarray,
                              lfs: pd.DataFrame,
                              label_vector: np.ndarray,
                              pred: Dict[str, Any],
                              label_idxs: List[int],
                              min_accuracy: float) -> List[int]:
        """Given the results of IWS execution, return the indexes of LFs that
        are selected according to the specified acquisition function."""
        if self.acquisition not in {'LSE', 'AS'}:
            raise ValueError(f'Unrecognised acquisition: {self.acquisition}')

        # Filter user-labelled indexes to those that were labelled as
        # useful.
        rule_idxs = [label_idx for label_idx in label_idxs
                     if label_vector[label_idx] == 1]

        # For LSE acquisition, add additional rules that are
        # predicted to be useful/accurate (if this selects multiple
        # rules for the same predicate, only keep the one with the
        # highest predicted accuracy).
        if self.acquisition == 'LSE':
            pred_df = pd.DataFrame({
                'rule': lfs.columns[pred['idxs']],
                'coverage': lfs.sum(axis=0)[pred['idxs']],
                'idx': pred['idxs'],
                'mean': pred['mean'],
            })
            pred_df['predicate_key'] = pred_df['rule'].apply(lambda rule: rule.predicate_key)
            predicate_to_max_pred = pred_df.groupby('predicate_key')['mean'].max().to_dict()
            pred_df['predicate_max_mean'] = pred_df['predicate_key'].map(predicate_to_max_pred)
            # Limit pred_df based on min_accuracy and whether it is
            # the best accuracy for a given predicate.
            pred_df = pred_df[(pred_df['mean'] >= pred_df['predicate_max_mean']) &
                              (pred_df['mean'] >= min_accuracy)]
            # LSE-ac: Limit to top 100 (the default value used in the
            # IWS paper) additional rules in terms of tradeoff between
            # coverage and accuracy (using a generalisation of the
            # original formula that accounts for multi-class).
            pred_df['metric'] = (pred_df['mean'] - (1 / classes.shape[0])) * pred_df['coverage']
            rule_idxs += pred_df.nlargest(100, columns='metric', keep='first')['idx'].tolist()

        # Ensure we have no duplicates.
        assert len(rule_idxs) == len(set(rule_idxs))
        return rule_idxs

    def generate_ruleset(self,
                         classes: np.array,
                         train_binary_features: pd.DataFrame,
                         train_y: pd.Series,
                         seed_rules: List[Rule],
                         rngseed: int,
                         interaction_count: int) -> Dict[str, Any]:
        min_accuracy = self.get_min_accuracy(classes)

        setup_start_time = time()
        train_binary_features = train_binary_features.loc[:, train_binary_features.mean() >= self.min_cov_ratio]

        possible_rules = self.get_possible_rules(classes, train_binary_features, train_y)
        lfs = self.get_lfs_for_rules(possible_rules, train_binary_features)
        initial_labels = {
            lfs.columns.get_loc(seed_rule): 1
            for seed_rule in seed_rules
        }
        log('Init IWS')
        LFfeatures = self.get_lf_features(classes, lfs)
        iws = InteractiveWeakSupervision(
            LFs=lfs.to_numpy(),
            LFfeatures=LFfeatures,
            initial_labels=initial_labels,
            oracle_func=self.get_oracle_func(lfs, train_y, min_accuracy=min_accuracy),
            acquisition=self.acquisition,
            rngseed=rngseed,
            nrandom_init=4,
        )
        setup_end_time = time()

        iws.run_experiments(num_iter=interaction_count)
        label_vector, label_sequence, time_sequence, pred_sequence = iws.rawdatadict[1]

        assert len(label_sequence) == min(lfs.shape[1], (interaction_count + len(initial_labels)))
        initial_label_idxs = label_sequence[:len(initial_labels)]
        interaction_label_idxs = label_sequence[len(initial_labels):]
        assert len(interaction_label_idxs) == len(time_sequence) == len(pred_sequence)

        ruleset = {
            'lfs': lfs,
            'setup_wall_seconds': (setup_end_time - setup_start_time),
            'interaction_label_idxs': interaction_label_idxs,
            'interaction_results': [
                {
                    'lf_idxs': self.get_iteration_lf_idxs(
                        classes=classes,
                        lfs=lfs,
                        label_vector=label_vector,
                        pred=pred_sequence[interaction_index],
                        label_idxs=[
                            *initial_label_idxs,
                            *interaction_label_idxs[:interaction_index]
                        ],
                        min_accuracy=min_accuracy,
                    ),
                    'wall_seconds': time_sequence[interaction_index],
                }
                for interaction_index in range(len(interaction_label_idxs))
            ],
        }
        return ruleset

    def get_rule_df(self, ruleset: Dict[str, Any], interaction_count: int) -> pd.DataFrame:
        max_interactions = min(interaction_count, len(ruleset['interaction_results']))
        interaction_result = ruleset['interaction_results'][max_interactions - 1]
        return self.lfs_to_rule_df(ruleset['lfs'].iloc[:, interaction_result['lf_idxs']])

    def get_elapsed_wall_seconds(self, ruleset: Dict[str, Any], interaction_count: int) -> float:
        elapsed_wall_seconds = ruleset['setup_wall_seconds']
        for interaction_ruleset in ruleset['interaction_results'][:interaction_count]:
            elapsed_wall_seconds += interaction_ruleset['wall_seconds']
        return elapsed_wall_seconds

    def get_extras(self, ruleset: Dict[str, Any], interaction_count: int) -> Dict[str, Any]:
        interaction_label_idxs = ruleset['interaction_label_idxs'][:interaction_count]
        return {
            'interaction_label_lfs': ruleset['lfs'].iloc[:, interaction_label_idxs],
        }

    def supports_partial_interactions(self) -> bool:
        return True

    @abstractmethod
    def get_raw_lf_feature_rows(self, classes: np.array, lfs: pd.DataFrame) -> Union[np.array, np.matrix]:
        """Each output row should represent a LF column in lfs."""
        pass


class IWSBinaryRG(IWSRG):
    """Uses original IWS representation for LFs: Entry for each LF and row
    with binary class labels represented by -1/1 and abstention by
    0."""

    def get_raw_lf_feature_rows(self, classes: np.array, lfs: pd.DataFrame) -> np.array:
        if len(classes) != 2:
            raise ValueError('BinaryIWS only supports 2 classes.')
        class_to_value = {classes[0]: 1, classes[1]: -1}
        return lfs.apply(
            lambda column: column.mask(column, class_to_value[column.name.target_class]),
            axis=0,
        ).T.to_numpy()


class IWSMultiRG(IWSRG):
    """Uses a multi-class representation for LFs: Entry for each LF, row,
    and class: with 1 when the LF assigns the given class, 0 for
    abstention, and -1 otherwise."""

    def get_raw_lf_feature_rows(self, classes: np.array, lfs: pd.DataFrame) -> np.matrix:
        instance_count = lfs.shape[0]
        lf_count = lfs.shape[1]

        row_idx_arrays = []
        col_idx_arrays = []
        data_arrays = []
        for row_idx, lf_column in enumerate(lfs.columns):
            covered_instance_idxs = lfs[lf_column].to_numpy().nonzero()[0]
            covered_count = covered_instance_idxs.shape[0]
            for class_idx, target_class in enumerate(classes):
                data_value = 1 if lf_column.target_class == target_class else -1
                row_idx_arrays.append(np.repeat(row_idx, covered_count))
                col_idx_arrays.append(covered_instance_idxs + (instance_count * class_idx))
                data_arrays.append(np.repeat(data_value, covered_count))

        row_idxs = np.concatenate(row_idx_arrays)
        col_idxs = np.concatenate(col_idx_arrays)
        data = np.concatenate(data_arrays)
        return csr_matrix((data, (row_idxs, col_idxs)),
                          shape=(lf_count, instance_count * classes.shape[0]))


class IWSDistinctRG(IWSRG):
    """Uses a single LF per feature that assigns the majority class (by
    assuming the user can choose the correct label) The LF
    representation has an entry for each LF and row with 1 for
    labelled instances and 0 for abstention."""

    def get_possible_rules(self,
                           classes: np.array,
                           train_binary_features: pd.DataFrame,
                           train_y: pd.Series) -> List[Rule]:
        """Return rules for an LF space of simple conditions of the given
        binary features returning the majority class."""
        def get_rule_class(feature_name: str) -> str:
            rule_covered_y = cast(pd.Series, train_y[train_binary_features[feature_name]])
            return rule_covered_y.mode().iloc[0]

        return [
            make_single_feature_rule(get_rule_class(feature_name), feature_name)
            for feature_name in train_binary_features.columns
        ]

    def get_raw_lf_feature_rows(self, classes: np.array, lfs: pd.DataFrame) -> np.ndarray:
        return lfs.astype(int).T.to_numpy()
