import numpy as np
from scipy import sparse

SNORKEL_ABSTAIN = -1


def build_snorkel_L(L, classes):
  """
  Given L containing names from classes and abstain values, return a
  snorkel-compatible version with integers replacing class names and
  the special SNORKEL_ABSTAIN value.
  """
  # Any entries in L that do not match a known class will be treated as an abstain.
  snorkel_L = np.full(L.shape, fill_value=SNORKEL_ABSTAIN)
  for class_idx, target_class in enumerate(classes):
    snorkel_L[L == target_class] = class_idx
  return snorkel_L.astype(int)


class Verifier(object):
    """
    A class for the Snorkel Model Verifier
    """

    def __init__(self, L_train, L_val, val_ground, classes, has_snorkel=True):
        self.L_train = L_train
        self.L_val = L_val
        self.val_ground = val_ground
        self.classes = classes

        self.snorkel_L_train = build_snorkel_L(self.L_train, self.classes)
        self.snorkel_L_val = build_snorkel_L(self.L_val, self.classes)

        self.has_snorkel = has_snorkel

    def train_gen_model(self,deps=False,grid_search=False):
        """
        Calls appropriate generative model
        """
        # CHANGED: Use new Snorkel syntax.
        from snorkel.labeling.model import LabelModel
        cardinality = len(self.classes)
        gen_model = LabelModel(cardinality=cardinality, verbose=True)
        gen_model.fit(self.snorkel_L_train, class_balance=(np.ones(cardinality) / cardinality), seed=0)
        self.gen_model = gen_model

    def assign_marginals(self):
        """
        Assigns probabilistic labels for train and val sets
        """
        self.train_marginals = self.gen_model.predict_proba(self.snorkel_L_train)
        self.val_marginals = self.gen_model.predict_proba(self.snorkel_L_val)
        #print 'Learned Accuracies: ', odds_to_prob(self.gen_model.w)

    def find_vague_points(self,gamma=0.2,b=0.5):
        """
        Find val set indices where marginals are within thresh of b
        """
        # CHANGED: gamma is in range [0, 1], so multiply gamma by b
        val_idx = np.where(np.abs(self.val_marginals - b) <= (gamma * b))
        return val_idx[0]
