import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

from .synthesizer import Synthesizer, ABSTAIN_LABEL
from .verifier import Verifier, SNORKEL_ABSTAIN

class HeuristicGenerator(object):
    """
    A class to go through the synthesizer-verifier loop
    """

    def __init__(self, train_primitive_matrix, val_primitive_matrix,
                 val_ground):
        """
        Initialize HeuristicGenerator object

        b: class prior of most likely class (TODO: use somewhere)
        beta: threshold to decide whether to abstain or label for heuristics
        gamma: threshold to decide whether to call a point vague or not
        """
        # CHANGED: Use a LabelEncoder to support multi-class string class labels.
        le = LabelEncoder().fit(val_ground)

        self.train_primitive_matrix = train_primitive_matrix
        self.val_primitive_matrix = val_primitive_matrix
        self.val_ground = le.transform(val_ground).astype(np.float64)
        self.class_labels = le.classes_
        self.classes = np.array(range(len(le.classes_))).astype(np.float64)
        self.cardinality = len(self.classes)
        self.b = 1 / self.cardinality

        self.vf = None
        self.syn = None
        self.hf = []
        self.feat_combos = []

    def apply_heuristics(self, heuristics, primitive_matrix, feat_combos, beta_opt):
        """
        Apply given heuristics to given feature matrix X and abstain by beta

        heuristics: list of pre-trained logistic regression models
        feat_combos: primitive indices to apply heuristics to
        beta: best beta value for associated heuristics
        """

        def marginals_to_labels(hf,X,beta):
            probs = hf.predict_proba(X)
            confidences = np.max(probs, axis=1)
            labels = hf.classes_[np.argmax(probs, axis=1)]
            labels_cutoff = labels
            labels_cutoff[confidences < (self.b + beta)] = ABSTAIN_LABEL
            return labels_cutoff

        L = np.stack([
            marginals_to_labels(hf,primitive_matrix[:,feat_combos[i]],beta_opt[i])
            for i,hf in enumerate(heuristics)
        ], axis=-1)
        assert L.shape == (np.shape(primitive_matrix)[0],len(heuristics))
        return L

    def prune_heuristics(self,heuristics,feat_combos,keep=1):
        """
        Selects the best heuristic based on Jaccard Distance and Reliability Metric

        keep: number of heuristics to keep from all generated heuristics
        """

        def calculate_jaccard_distance(num_labeled_total, num_labeled_L):
            scores = np.zeros(np.shape(num_labeled_L)[1])
            for i in range(np.shape(num_labeled_L)[1]):
                scores[i] = np.sum(np.minimum(num_labeled_L[:,i],num_labeled_total))/np.sum(np.maximum(num_labeled_L[:,i],num_labeled_total))
            return 1-scores

        L_val = np.array([])
        L_train = np.array([])
        beta_opt = np.array([])
        max_cardinality = len(heuristics)
        for i in range(max_cardinality):
            #Note that the LFs are being applied to the entire val set though they were developed on a subset...
            beta_opt_temp = self.syn.find_optimal_beta(heuristics[i], self.val_primitive_matrix, feat_combos[i], self.val_ground)
            L_temp_val = self.apply_heuristics(heuristics[i], self.val_primitive_matrix, feat_combos[i], beta_opt_temp)
            L_temp_train = self.apply_heuristics(heuristics[i], self.train_primitive_matrix, feat_combos[i], beta_opt_temp)

            beta_opt = np.append(beta_opt, beta_opt_temp)
            if i == 0:
                L_val = np.append(L_val, L_temp_val) #converts to 1D array automatically
                L_val = np.reshape(L_val,np.shape(L_temp_val))
                L_train = np.append(L_train, L_temp_train) #converts to 1D array automatically
                L_train = np.reshape(L_train,np.shape(L_temp_train))
            else:
                L_val = np.concatenate((L_val, L_temp_val), axis=1)
                L_train = np.concatenate((L_train, L_temp_train), axis=1)

        #Use F1 trade-off for reliability
        acc_cov_scores = [f1_score(self.val_ground, L_val[:,i], average='micro') for i in range(np.shape(L_val)[1])]
        acc_cov_scores = np.nan_to_num(acc_cov_scores)

        if self.vf != None:
            #Calculate Jaccard score for diversity
            # CHANGED: Use any instead of sum to better reflect intent
            # (the behaviour should not change, as they are
            # treated as booleans when calculating jaccard distance)
            train_num_labeled = np.any(self.vf.L_train != ABSTAIN_LABEL, axis=1)
            jaccard_scores = calculate_jaccard_distance(train_num_labeled,(L_train != ABSTAIN_LABEL))
        else:
            jaccard_scores = np.ones(np.shape(acc_cov_scores))

        #Weighting the two scores to find best heuristic
        combined_scores = 0.5*acc_cov_scores + 0.5*jaccard_scores
        sort_idx = np.argsort(combined_scores)[::-1][0:keep]
        return sort_idx


    def run_synthesizer(self, max_cardinality=1, idx=None, keep=1, model='lr'):
        """
        Generates Synthesizer object and saves all generated heuristics

        max_cardinality: max number of features candidate programs take as input
        idx: indices of validation set to fit programs over
        keep: number of heuristics to pass to verifier
        model: train logistic regression ('lr') or decision tree ('dt')
        """
        if idx == None:
            primitive_matrix = self.val_primitive_matrix
            ground = self.val_ground
        else:
            primitive_matrix = self.val_primitive_matrix[idx,:]
            ground = self.val_ground[idx]


        #Generate all possible heuristics
        self.syn = Synthesizer(primitive_matrix, ground, b=self.b)

        #Un-flatten indices
        def index(a, inp):
            i = 0
            remainder = 0
            while inp >= 0:
                remainder = inp
                inp -= len(a[i])
                i+=1
            try:
                return a[i-1][remainder] #TODO: CHECK THIS REMAINDER THING WTF IS HAPPENING
            except:
                import pdb; pdb.set_trace()

        #Select keep best heuristics from generated heuristics
        hf, feat_combos = self.syn.generate_heuristics(model, max_cardinality)
        sort_idx = self.prune_heuristics(hf,feat_combos, keep)
        for i in sort_idx:
            self.hf.append(index(hf,i))
            self.feat_combos.append(index(feat_combos,i))

        #create appended L matrices for validation and train set
        beta_opt = self.syn.find_optimal_beta(self.hf, self.val_primitive_matrix, self.feat_combos, self.val_ground)
        self.L_val = self.apply_heuristics(self.hf, self.val_primitive_matrix, self.feat_combos, beta_opt)
        self.L_train = self.apply_heuristics(self.hf, self.train_primitive_matrix, self.feat_combos, beta_opt)

    def run_verifier(self):
        """
        Generates Verifier object and saves marginals
        """
        ###THIS IS WHERE THE SNORKEL FLAG IS SET!!!!
        self.vf = Verifier(self.L_train, self.L_val, self.val_ground,
                           classes=self.classes, has_snorkel=True)
        self.vf.train_gen_model()
        self.vf.assign_marginals()

    def gamma_optimizer(self,marginals):
        """
        Returns the best gamma parameter for abstain threshold given marginals

        marginals: confidences for data from a single heuristic
        """
        m = len(self.hf)
        # CHANGED: Using m+1 as in paper
        # orig_gamma is in range [0, 0.5]
        orig_gamma = 0.5-(1/((m + 1)**(3/2.)))
        # scale to [0, 1], for use in multi-class scenarios
        gamma = orig_gamma * 2
        return gamma

    def find_feedback(self):
        """
        Finds vague points according to gamma parameter

        self.gamma: confidence past 0.5 that relates to a vague or incorrect point
        """
        #TODO: flag for re-classifying incorrect points
        #incorrect_idx = self.vf.find_incorrect_points(b=self.b)

        gamma_opt = self.gamma_optimizer(self.vf.val_marginals)
        #gamma_opt = self.gamma
        vague_idx = self.vf.find_vague_points(b=self.b, gamma=gamma_opt)
        incorrect_idx = vague_idx
        self.feedback_idx = list(set(list(np.concatenate((vague_idx,incorrect_idx)))))
