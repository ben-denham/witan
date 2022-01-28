"""
Parts of this code sourced from:
https://github.com/benbo/interactive-weak-supervision/blob/acb7603eb003d3857e21957cb22ae15f20061d51/torchmodels.py,
and https://github.com/benbo/interactive-weak-supervision/blob/acb7603eb003d3857e21957cb22ae15f20061d51/IWS.ipynb used under license:

MIT License

Copyright (c) 2020 Benedikt Boecking

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
import torch

from . import transformers as tf
from ..datasets import Dataset
from ..utils import torch_weight_reset, torch_seed_reset, KeyHashable


def build_feature_transformer(dataset: Dataset) -> TransformerMixin:
    """Construct a transformer that transforms dfs from a given Dataset to
    a vectorized text features array."""
    return Pipeline([
        ('select_features', tf.SelectFeatureSubsetDataFrameTransformer(
            list(dataset.text_features))),
        ('text', tf.TransformFeatureSubsetDataFrameTransformer(
            list(dataset.text_features),
            # Same as:
            # https://github.com/benbo/interactive-weak-supervision/blob/main/IWS.ipynb
            tf.DataFrameVectorizer(CountVectorizer(
                strip_accents='ascii',
                stop_words='english',
                ngram_range=(1, 1),
                analyzer='word',
                max_df=0.9,
                min_df=10,
                max_features=None,
                vocabulary=None,
                binary=False,
            ))
        )),
    ])


def binarize_features(features_array: np.ndarray) -> np.ndarray:
    """Convert vectorized text features into binary bag-of-words features."""
    return features_array > 0


class Classifier(ABC, KeyHashable):
    """Base class for classifiers."""

    @abstractmethod
    def predict_probs(self,
                      classes: np.array,
                      covered_train_features: pd.DataFrame,
                      test_features: pd.DataFrame,
                      covered_train_prob_labels: np.ndarray) -> np.ndarray:
        """Train a classifier on given covered_train_features and
        covered_train_prob_labels, and return probabilistic classifications
        for test_features."""
        pass


class AnnClf(Classifier):
    """Artificial Neural Network classifier based on model from IWS codebase."""

    def get_feature_transformer(self,
                                row_count: int,
                                feature_count: int) -> TransformerMixin:
        """Return transformer to prepare features for the classifier."""
        # Do not perform SVD with fewer rows than components.
        svd_components = min(row_count, 300)
        steps = [
            # torchmodel expects float32
            ('type', tf.DataFrameDtypeTransformer(np.float32)),
            ('sparse_matrix', tf.SparseMatrixTransformer()),
        ]
        if feature_count > svd_components:
            steps.append(
                ('svd', TruncatedSVD(
                    n_components=svd_components,
                    n_iter=20,
                    random_state=42,
                )))
        else:
            steps.append(
                ('dense', tf.DenseTransformer()),
            )
        return Pipeline(steps)

    def predict_probs(self,
                      classes: np.array,
                      covered_train_features: pd.DataFrame,
                      test_features: pd.DataFrame,
                      covered_train_prob_labels: np.array):
        class_balance = covered_train_prob_labels.sum(axis=0)
        assert class_balance.shape == classes.shape

        # Short circuit if there is only a single class in the
        # training set.
        if (class_balance > 0).sum() == 1:
            sole_class_idx = np.argmax(class_balance)
            probs = np.zeros((test_features.shape[0], classes.shape[0]))
            probs[:, sole_class_idx] = 1
            return probs

        # Find and remove feature columns that only contain a constant
        # value (avoids nan issues in SVD transformation).
        nonconstant_columns = covered_train_features.columns[covered_train_features.nunique() > 1]
        covered_train_features = covered_train_features[nonconstant_columns]
        test_features = test_features[nonconstant_columns]

        feature_transformer = self.get_feature_transformer(
            row_count=covered_train_features.shape[0],
            feature_count=covered_train_features.shape[1],
        )
        covered_train_features = feature_transformer.fit_transform(covered_train_features)
        test_features = feature_transformer.transform(test_features)

        torch.cuda.empty_cache()
        model = TorchMLP(h_sizes=[covered_train_features.shape[1], 20, 20],
                         activations=[torch.nn.ReLU(), torch.nn.ReLU()],
                         cardinality=covered_train_prob_labels.shape[1],
                         optimizer='Adam', nepochs=250)

        device = 'cpu'
        if 'cuda' in device:
            tdevice = torch.device(device)
            model.model.to(tdevice)
            model.fit(covered_train_features,
                      covered_train_prob_labels.astype(np.float32),
                      device=tdevice)
            test_predictions = model.predict_proba(test_features, device=tdevice)
        else:
            model.fit(covered_train_features,
                      covered_train_prob_labels.astype(np.float32))
            test_predictions = model.predict_proba(test_features)
        return test_predictions


class FeedforwardFlexible(torch.nn.Module):
    """Neural network architecture from IWS codebase."""

    def __init__(self, h_sizes, activations, cardinality):
        super(FeedforwardFlexible, self).__init__()

        self.layers = torch.nn.ModuleList()
        for k in range(len(h_sizes)-1):
            self.layers.append(torch.nn.Linear(h_sizes[k], h_sizes[k+1]))
            self.layers.append(activations[k])

        # CHANGED FROM IWS: Using a multi-output softmax instead of a
        # single-output sigmoid to support multi-class classification.
        self.layers.append(torch.nn.Linear(h_sizes[-1], cardinality))
        self.layers.append(torch.nn.Softmax(dim=1))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TorchMLP:
    """Multi-layer Perceptron from IWS codebase."""

    def __init__(self, cardinality, h_sizes=[150, 10, 10],
                 activations=[torch.nn.ReLU(), torch.nn.ReLU()],
                 optimizer='Adam', optimparams={}, nepochs=200):
        self.model = FeedforwardFlexible(h_sizes, activations, cardinality).float()
        self.optimizer = optimizer
        if optimizer == 'Adam':
            if optimparams:
                self.optimparams = optimparams
            else:
                self.optimparams = {'lr': 1e-3, 'weight_decay': 1e-4}

        self.epochs = nepochs

    def fit(self, X, Y, sample_weights=None, device=None):
        torch_seed_reset()

        tinput = torch.from_numpy(X)
        target = torch.from_numpy(Y)
        if device is not None:
            tinput = tinput.to(device)
            target = target.to(device)
        tweights = None
        if sample_weights is not None:
            tweights = torch.from_numpy(sample_weights.reshape(-1, 1))
            if device is not None:
                tweights = tweights.to(device)

        criterion = torch.nn.BCELoss(reduction='none')
        self.model.apply(torch_weight_reset)

        trainX, trainy = tinput, target
        trainweight = None
        if tweights is not None:
            trainweight = tweights

        if self.optimizer == 'LBFGS':
            optimizer = torch.optim.LBFGS(self.model.parameters(),
                                          lr=1,
                                          max_iter=400,
                                          max_eval=15000,
                                          tolerance_grad=1e-07,
                                          tolerance_change=1e-04,
                                          history_size=10,
                                          line_search_fn=None)

            def closure():
                optimizer.zero_grad()
                mout = self.model(trainX)
                closs = criterion(mout, trainy)
                if tweights is not None:
                    closs = torch.mul(closs, trainweight).mean()
                else:
                    closs = closs.mean()

                closs.backward()
                return closs
            # only take one step (one epoch)
            optimizer.step(closure)
        else:
            optimizer = None
            if self.optimizer == 'Adam':
                optimizer = torch.optim.Adam(self.model.parameters(), **self.optimparams)
            elif self.optimizer == 'SGD':
                optimizer = torch.optim.SGD(self.model.parameters(), **self.optimparams)
            lastloss = None
            tolcount = 0
            for nep in range(self.epochs):

                out = self.model(trainX)
                loss = criterion(out, trainy)
                if tweights is not None:
                    loss = torch.mul(loss, trainweight).mean()
                else:
                    loss = loss.mean()

                # early stopping
                if lastloss is None:
                    lastloss = loss
                else:
                    if lastloss-loss < 1e-04:
                        tolcount += 1
                    else:
                        tolcount = 0
                    if tolcount > 9:
                        break

                optimizer.zero_grad()
                loss.backward()

                optimizer.step()

    def predict(self, Xtest, device=None):
        preds = self.predict_proba(Xtest, device)
        return preds.argmax(axis=1).astype(int)

    def predict_proba(self, Xtest, device=None):
        with torch.no_grad():
            tXtest = torch.from_numpy(Xtest)
            if device is not None:
                tXtest = tXtest.to(device)
                preds = self.model(tXtest).data.cpu().numpy()
            else:
                preds = self.model(tXtest).data.numpy()
        return preds
