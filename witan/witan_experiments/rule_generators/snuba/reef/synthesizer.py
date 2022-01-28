import numpy as np
import itertools

from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Use an integer label that will not conflict with the LabelEncoder
ABSTAIN_LABEL = -1


class Synthesizer(object):
    """
    A class to synthesize heuristics from primitives and validation labels
    """
    def __init__(self, primitive_matrix, val_ground,b=0.5):
        """
        Initialize Synthesizer object

        b: class prior of most likely class
        beta: threshold to decide whether to abstain or label for heuristics
        """
        self.val_primitive_matrix = primitive_matrix
        self.val_ground = val_ground
        self.p = np.shape(self.val_primitive_matrix)[1]
        self.b=b

    def generate_feature_combinations(self, cardinality=1):
        """
        Create a list of primitive index combinations for given cardinality

        max_cardinality: max number of features each heuristic operates over
        """
        primitive_idx = range(self.p)
        feature_combinations = []

        for comb in itertools.combinations(primitive_idx, cardinality):
            feature_combinations.append(comb)

        return feature_combinations

    def fit_function(self, comb, model):
        """
        Fits a single logistic regression or decision tree model

        comb: feature combination to fit model over
        model: fit logistic regression or a decision tree
        """
        X = self.val_primitive_matrix[:,comb]
        if np.shape(X)[0] == 1:
            X = X.reshape(-1,1)

        # fit decision tree or logistic regression or knn
        if model == 'dt':
            dt = DecisionTreeClassifier(max_depth=len(comb))
            dt.fit(X,self.val_ground)
            return dt

        elif model == 'lr':
            lr = LogisticRegression()
            lr.fit(X,self.val_ground)
            return lr

        elif model == 'nn':
            nn = KNeighborsClassifier(algorithm='kd_tree')
            nn.fit(X,self.val_ground)
            return nn

    def generate_heuristics(self, model, max_cardinality=1):
        """
        Generates heuristics over given feature cardinality

        model: fit logistic regression or a decision tree
        max_cardinality: max number of features each heuristic operates over
        """
        #have to make a dictionary?? or feature combinations here? or list of arrays?
        feature_combinations_final = []
        heuristics_final = []
        for cardinality in range(1, max_cardinality+1):
            feature_combinations = self.generate_feature_combinations(cardinality)

            heuristics = []
            for i,comb in enumerate(feature_combinations):
                heuristics.append(self.fit_function(comb, model))

            feature_combinations_final.append(feature_combinations)
            heuristics_final.append(heuristics)

        return heuristics_final, feature_combinations_final

    def beta_optimizer(self, labels, confidences, ground):
        """
        Returns the best beta parameter for abstain threshold given marginals
        Uses F1 score that maximizes the F1 score

        confidences: confidences for data from a single heuristic
        """

        #Set the range of beta params
        #0.25 instead of 0.0 as a min makes controls coverage better
        # CHANGED: Generalise for multi-class by makin space relative to self.b
        max_beta = 1 - self.b
        beta_params = np.linspace(max_beta / 2, max_beta * 0.8, 10)

        f1 = []

        for beta in beta_params:
            labels_cutoff = np.copy(labels)
            labels_cutoff[confidences < (self.b + beta)] = ABSTAIN_LABEL
            f1.append(f1_score(ground, labels_cutoff, average='weighted'))

        f1 = np.nan_to_num(f1)
        return beta_params[np.argsort(np.array(f1))[-1]]


    def find_optimal_beta(self, heuristics, X, feat_combos, ground):
        """
        Returns optimal beta for given heuristics

        heuristics: list of pre-trained logistic regression models
        X: primitive matrix
        feat_combos: feature indices to apply heuristics to
        ground: ground truth associated with X data
        """

        beta_opt = []
        for i,hf in enumerate(heuristics):
            probs = hf.predict_proba(X[:,feat_combos[i]])
            confidences = np.max(probs, axis=1)
            labels = hf.classes_[np.argmax(probs, axis=1)]
            # CHANGED: Optimize based on confidences instead of
            # marginals for multi-class support. The results should be
            # identical for the binary case.
            beta_opt.append(self.beta_optimizer(labels, confidences, ground))
        return beta_opt
