"""
Original code sourced from:
https://github.com/benbo/interactive-weak-supervision/blob/acb7603eb003d3857e21957cb22ae15f20061d51/iws.py,
https://github.com/benbo/interactive-weak-supervision/blob/acb7603eb003d3857e21957cb22ae15f20061d51/utils.py, and
https://github.com/benbo/interactive-weak-supervision/blob/acb7603eb003d3857e21957cb22ae15f20061d51/torchmodels.py used under license:

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

from time import time
import numpy as np
from sklearn.utils import resample
import torch
import torch.multiprocessing

from ...utils import log, torch_weight_reset, torch_seed_reset, drop_keys


class InteractiveWeakSupervision:
    def __init__(self, LFs, LFfeatures, initial_labels, oracle_func,
                 acquisition='LSE', r=0.6, nrandom_init=4, g_inv=None,
                 straddle_z=1.96, ensemblejobs=1, rngseed=1):
        """
            Code to collect user or oracle feedback for
            Interactive Weak Supervision.

            Parameters
            ----------
            LFs : sparse matrix, shape (n_samples, p_LFs)
                Sparse matrix of generated LFs (boolean matrix indicating LF firing, not class value)
            LFfeatures: array, shape (p_LFs, d')
                Features of dimension d' for each of the p generated LFs
            initial_labels : dict
                A dictionary containing indices of labeling functions in the  LFs matrix
                that have some labels for {LFidx : label}, to initialize the algorithm.
            oracle_func : function
                Function accepts an LF index and returns a boolean
                if that LF is believed to be useful.
            ensemblejobs : int, default=1
                number threads to parallelize ensemble
            nrandom_init : int, default = None
                Number of random queries to initialize IWS
            g_inv : callable, default=None
                A callable that takes in a matrix of p(u=1) estimates and returns the mapping to alpha (the LF accuracy)
                It is the inverse of the g function, which maps from latent LF accuracy alpha_j to v_j and thus
                g_inv(v_j) = alpha_j. The default assumes g to be the identity.
            acquisition : str, one of 'LSE','AS','random'
                Acquisition function to use. Choice depends on final set of LF we want to estimate.
                Choose 'AS' if only LFs that are inspected by users will be used in label model.
            """
        # generated LFs
        self.LFs = LFs
        # LF features for generated LFs
        self.X = LFfeatures.astype(np.float32)
        self.N, self.M = self.LFs.shape
        self.idxs = np.arange(self.M)
        if self.X.shape[0] != self.M:
            raise ValueError('Number of LFs in LF features does not equal number of LFs in variableLFs')

        # initialization
        self.maxiter = None
        self.counter = None
        if nrandom_init is None:
            self.nrandom_init = len(initial_labels.keys())
        else:
            self.nrandom_init = nrandom_init
        self.rng = np.random.RandomState(rngseed)

        #  dictionary to store data from each repeated run
        self.rawdatadict = {}
        self.runidx = 1  # init index that keeps track of number of repeated experiments
        self.acquisition = acquisition
        # multiplication factor
        self.straddle_z = straddle_z
        if 0.5 <= r <= 1:
            self.straddle_threshold = r
        else:
            ValueError('Choose r in [0.5,1.0]')

        if g_inv is None:
            self.g_inv = lambda x: x  # define g to be the identity
        else:
            self.g_inv = g_inv

        # Check acquisition function setting
        if acquisition == 'LSE':
            self.acquisitionfunc = self.straddling_threshold_proba
        elif acquisition == 'AS':
            self.acquisitionfunc = self.active_search_greedy
        elif acquisition == 'random':
            self.acquisitionfunc = self.random_acquisition
        else:
            errmessage = 'Acquisition not implemented. Choose from: LSE, AS, random'
            raise NotImplementedError(errmessage)

        # set up ensemble
        self.model = BaggingWrapperTorch(n_estimators=50,
                                         njobs=ensemblejobs,
                                         nfeatures=LFfeatures.shape[1],
                                         rng=self.rng)

        # We will not model uncertainty about response
        self.oracle_func = oracle_func

        # set up initial labels
        # process initial labels
        self.labeldict = {}  # duplicate info but useful for faster lookup
        self.labelvector = np.ones(self.M, dtype=np.float32) * np.inf
        self.labelsequence = []
        self.timesequence = []
        self.predsequence = []
        self.initial_labels = initial_labels  # so we can save this info

        for idx, val in initial_labels.items():
            self.labelsequence.append(idx)
            self.labeldict[idx] = 1.0
            self.labelvector[idx] = 1.0

        # handle empty LFs (they are not useful)
        colsums = self.LFs.sum(0)
        colsums = np.asarray(colsums).flatten()
        idxs = np.where(colsums == 0)[0]
        if len(idxs) > 0:
            for idx in idxs:
                self.labelvector[idx] = 0.0
                self.labeldict[idx] = 0.0

        # Initialise model output caching variables
        self._cache_model_train_test_result = None
        self._cache_model_train_test_idxbool = None

    def straddling_threshold_proba(self):
        # straddling with scores
        # get value of function inferred by model
        pred = self.model_train_test()
        pred_idxs = pred['idxs']

        # 1.96 * std-dev - |prediction - threshold|
        acqusitionfunction = self.straddle_z * pred['dev'] - np.abs(pred['mean'] - self.straddle_threshold)

        idx = pred_idxs[np.argmax(acqusitionfunction)]

        return idx

    def active_search_greedy(self):
        # get value of function inferred by model
        pred = self.model_train_test()
        pred_idxs = pred['idxs']

        idx = pred_idxs[np.argmax(pred['mean'])]
        return idx

    def random_acquisition(self):
        # pick random LF
        idxsbool = self.labelvector == np.inf
        idx = self.rng.choice(self.idxs[idxsbool])
        return idx

    def model_train_test(self):
        # get samples we have labels for
        idxbool = self.labelvector != np.inf
        # Check if we can use cache.
        if (
            (self._cache_model_train_test_result is None) or
            (not np.array_equal(idxbool, self._cache_model_train_test_idxbool))
        ):
            Y = self.labelvector[idxbool]
            X = self.X[idxbool]
            Xtest = self.X[~idxbool]

            if X.shape[0] == 0:
                A = np.zeros((Xtest.shape[0], 1))
            else:
                # fit
                self.model.fit(X, Y)
                # return scores on labeling functions we don't have feedback for
                # also return the boolean index

                # predict returns mean and std for discrete distribution
                V = self.model.predict_raw(Xtest)  # matrix of p(u=1|Q_t)
                A = self.g_inv(V)  # use g_inv to map to latent LF accuracy

            self._cache_model_train_test_idxbool = idxbool
            self._cache_model_train_test_result = {
                'mean': A.mean(1),
                'dev': A.std(1),
                'idxs': self.idxs[~idxbool],
            }
        return self._cache_model_train_test_result

    def reset(self):
        self.rawdatadict[self.runidx] = (self.labelvector, self.labelsequence,
                                         self.timesequence, self.predsequence)
        self.runidx += 1
        self.labelsequence = []
        self.timesequence = []
        self.predsequence = []
        self.labeldict = {}  # duplicate info but useful for faster lookup
        self.labelvector = np.ones(self.M, dtype=np.float32) * np.inf

        self._cache_model_train_test_result = None
        self._cache_model_train_test_idxbool = None

        for idx, val in self.initial_labels.items():
            self.labelsequence.append(idx)
            self.labeldict[idx] = 1.0
            self.labelvector[idx] = 1.0
        # handle empty LFs (they are not useful)
        colsums = self.LFs.sum(0)
        colsums = np.asarray(colsums).flatten()
        idxs = np.where(colsums == 0)[0]
        if len(idxs) > 0:
            for idx in idxs:
                self.labelvector[idx] = 0.0
                self.labeldict[idx] = 0.0

    def run_experiments(self, num_iter):
        self.maxiter = num_iter
        self.counter = 0
        self.next_candidate()

    def next_candidate(self):
        iteration_start_time = time()
        if (self.counter >= self.maxiter) or (self.idxs[(self.labelvector == np.inf)].shape[0] == 0):
            self.reset()
            return

        log(f'Running IWS iteration: {self.counter + 1}')

        if self.counter < self.nrandom_init:
            # random during initialization
            idx = self.random_acquisition()
        else:
            # maximize acquisition function to get next candidate LF
            idx = self.acquisitionfunc()
        self.counter += 1

        # use oracle
        oracle_start_time = time()
        if self.oracle_func(idx):
            lbl = 1
        else:
            lbl = 0
        oracle_end_time = time()

        self.labeldict[idx] = lbl
        self.labelvector[idx] = lbl
        self.labelsequence.append(idx)
        if self.acquisition == 'LSE':
            self.predsequence.append(drop_keys(self.model_train_test(), 'dev'))
        else:
            self.predsequence.append(None)
        iteration_end_time = time()

        iteration_time = iteration_end_time - iteration_start_time
        oracle_time = oracle_end_time - oracle_start_time
        self.timesequence.append(iteration_time - oracle_time)

        self.next_candidate()


# TORCH MODEL FOR LF ACQUISITION

torch.multiprocessing.set_sharing_strategy('file_system')


class Feedforward(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.relu2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(self.hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        hidden1 = self.fc1(x)
        relu1 = self.relu1(hidden1)
        hidden2 = self.fc2(relu1)
        relu2 = self.relu2(hidden2)
        output = self.fc3(relu2)
        out = self.sigmoid(output)
        return out


def applypredict(args):
    model, Xtest = args
    return model(Xtest).data.numpy().flatten()


def applyfit(args):
    torch_seed_reset()

    # reset model
    model, ix, N, tinput, target, tweights, random_state, whichoptim, epochs, optimparams = args
    criterion = torch.nn.BCELoss(reduction='none')
    model.apply(torch_weight_reset)
    # select indexes
    train_ix = resample(ix, replace=True, n_samples=N, random_state=random_state)
    trainX, trainy = tinput[train_ix], target[train_ix]
    trainweight = None
    optimizer = None
    if tweights is not None:
        trainweight = tweights[train_ix]

    if whichoptim == 'LBFGS':
        optimizer = torch.optim.LBFGS(model.parameters(),
                                      lr=0.1,
                                      max_iter=400,
                                      max_eval=15000,
                                      tolerance_grad=1e-07,
                                      tolerance_change=1e-04,
                                      history_size=10,
                                      line_search_fn=None)

        def closure():
            optimizer.zero_grad()
            cout = model(trainX)
            closs = criterion(cout, trainy)
            if tweights is not None:
                closs = torch.mul(closs, trainweight).mean()
            else:
                closs = closs.mean()
            closs.backward()
            return closs

        lloss = 1.0
        cntr = 0
        while lloss > 0.1 and cntr < epochs:
            cntr += 1
            optimizer.step(closure)
            with torch.no_grad():
                out = model(trainX)
                lloss = criterion(out, trainy)
                if tweights is not None:
                    lloss = torch.mul(lloss, trainweight).mean()
                else:
                    lloss = lloss.mean()
                if lloss > 0.1:
                    model.apply(torch_weight_reset)
    else:
        if whichoptim == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), **optimparams)
        elif whichoptim == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), **optimparams)
        lastloss = None
        tolcount = 0
        for nep in range(epochs):
            out = model(trainX)
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


class BaggingWrapperTorch:
    def __init__(self, nfeatures=150, n_estimators=100, njobs=1,
                 optimizer='Adam', optimparams={}, nepochs=200,
                 rng=np.random):
        self.n_estimators = n_estimators
        self.device = torch.device("cpu")  # parallel ensemble only implemented on cpu for now
        self.members = [Feedforward(nfeatures, 10).float().to(self.device) for _ in range(n_estimators)]
        self.mpool = None
        self.njobs = njobs
        self.optimizer = optimizer
        self.rng = rng

        if optimizer == 'Adam':
            if optimparams:
                self.optimparams = optimparams
            else:
                self.optimparams = {'lr': 1e-3, 'weight_decay': 1e-4}

        self.epochs = nepochs
        if njobs > 1:
            import multiprocessing as mp
            ctx = mp.get_context('spawn')
            _ = applyfit
            self.mpool = ctx.Pool(self.njobs)

    def fit(self, X, Y, sample_weights=None):
        if np.any(np.isnan(Y)):
            raise ValueError('found nan in Y')
        random_states = self.rng.randint(0, 2**32 - 1, self.n_estimators)
        tinput = torch.from_numpy(X)
        target = torch.from_numpy(Y.reshape(-1, 1))
        tweights = None
        if sample_weights is not None:
            tweights = torch.from_numpy(sample_weights.reshape(-1, 1))
        N = len(X)
        ix = list(range(N))
        if self.njobs > 1:
            self.mpool.map(applyfit, list((model, ix, N, tinput, target, tweights, rstate,
                           self.optimizer, self.epochs, self.optimparams) for model, rstate in
                           zip(self.members, random_states)))

        else:
            for j, model in enumerate(self.members):
                applyfit((model, ix, N, tinput, target, tweights, random_states[j], self.optimizer,
                          self.epochs, self.optimparams))

    def predict_raw(self, Xtest):
        with torch.no_grad():
            n = Xtest.shape[0]
            tXtest = torch.from_numpy(Xtest)
            predictions = np.zeros((n, self.n_estimators))

            if self.njobs > 1:
                for i, pred in enumerate(self.mpool.map(applypredict, list((model, tXtest)
                                                                           for model in self.members))):
                    predictions[:, i] = pred
            else:
                for i, model in enumerate(self.members):
                    predictions[:, i] = model(tXtest).data.numpy().flatten()

            return predictions

    def predict(self, Xtest):
        predictions = self.predict_raw(Xtest)
        predictions[predictions > 0.5] = 1
        predictions[predictions <= 0.5] = 0
        return predictions.mean(1), predictions.std(1)

    def predict_proba(self, Xtest):
        predictions = self.predict_raw(Xtest)
        return predictions.mean(1), predictions.std(1)
