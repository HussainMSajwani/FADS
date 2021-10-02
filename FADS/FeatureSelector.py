from lassonet import LassoNetClassifier
#from pyHSICLasso import HSICLasso

from .input import get_dset
from .utils import which_GPU
#from .models import supervised_autoencoder
from .auc import calculate_auc

import matplotlib.pyplot as plt
from torch.cuda import empty_cache
import numpy as np
from functools import partial
from statsmodels.api import Logit, tools
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

class FeatureSelector:

    def __init__(self, dset):
        self.dset = dset
        self.model = None
        self.importance = []
        self.max_k = np.inf

    def top_k(self, k):
        """Returns top n SNPs the FeatureSelector thinks are most important

        Args:
            k (int): k

        Returns:
            numpy.array: index array of top k SNPs
        """
        if len(self.importance) == 0:
            self.train()
        if k <= self.max_k:
            return np.flip(np.argsort(self.importance))[:k]
        else:
            return None

    def manhattan(self):
        """Generate Manhattan plot using FeatureSelector
        """
        if not len(self.importance):
            self.train()

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

        groups = self.dset["ct"].groupby("causal")
        for is_causal, group in groups:
            plt.scatter(group.index, self.importance[group.index], label="causal" if is_causal else "not causal")
        plt.legend(title="causality")
        plt.xlabel("SNPs")
        plt.show()
    
    def train_svm(self, k):
        """Trains an SVM on top k SNPs and returns AUC

        Args:
            k (int): How many SNPs to use

        Returns:
            float: AUC of SVM
        """
        top_k = self.top_k(k)
        svm = SVC(probability=True)
        svm.fit(self.dset["X_train"][:, top_k]/2, self.dset["y_train"][:, 1])

        auc = calculate_auc(
            lambda X: svm.predict_proba(X)[:, 1],
            self.dset["X_test"][:, top_k],
            self.dset["y_test"][:, 1]
        )

        return auc

    def train_lr(self, k):
        """Trains a logistic regression model on top k SNPs and returns AUC

        Args:
            k (int): How many SNPs to use

        Returns:
            float: AUC
        """
        top_k = self.top_k(k)

        lr = LogisticRegression(max_iter=5000)
        lr.fit(self.dset["X_train"][:, top_k]/2, self.dset["y_train"][:, 1])

        auc = calculate_auc(
            lambda X: lr.predict_proba(X)[:, 1],
            self.dset["X_test"][:, top_k],
            self.dset["y_test"][:, 1]
        )

        return auc

    def n_causals_top_k(self, k):
        """returns number of causal SNPs in top k SNPs

        Args:
            k (int): number of SNPs to consider

        Returns:
            int: number of causals in top k
        """
        top_k = self.top_k(k)
        
        n_causals_caught = [snp in top_k for snp in self.dset["ct"].query("causal == 1").index]
        return np.sum(n_causals_caught)

    def save(self, path):
        with open(path + "/importance.csv"):
            np.save(path + "/importance.csv", self.importance)

    def train(self):
        pass

class LassoNet(FeatureSelector):

    def __init__(self, dset, architecture):
        super().__init__(dset)
        self.architecture = architecture

    def train(self):
        self.model = LassoNetClassifier(verbose=False, hidden_dims=self.architecture)
        
        self.ln_path = self.model.path(self.dset["X_train"]/2, self.dset["y_train"][:, 1])
        self.importance = self.model.feature_importances_.cpu().numpy()

        #del model
        empty_cache()

class pValue(FeatureSelector):

    def __init__(self, dset):
        super().__init__(dset)

    def single_pval(self, i, X, y):
        lr = Logit(y, tools.add_constant(X[:, i])).fit(disp=0, method="bfgs") 
        pvals = lr.pvalues
        if len(pvals) != 2:
            p = 1
        else:
            p = pvals[1]
        return p if not p == 0 else 1e-100

    def train(self):
        out = []
    #P = Pool(int(cpu_count()/2 - 1))
        pvals_iterable = map(partial(self.single_pval, X=self.dset["X_train"]/2, y=self.dset["y_train"][:, 1]), range(self.dset["X_train"].shape[1]))

        with tqdm(total = self.dset["X_train"].shape[1], desc="generating p values") as pbar:
            for p_val in pvals_iterable:
                out.append(-np.log10(p_val))
                pbar.update()

        self.importance = np.array(out)

class SAE(FeatureSelector):
    
    def __init__(self, dset, sizes):
        super().__init__(dset)
        hist, model = supervised_autoencoder(dset["X_train"], dset["y_train"], sizes=sizes, activation="relu", l1=3e-6)
        
        weights = model.layers[1].get_weights()[0]
        self.importance = np.sum(np.abs(weights), axis=1)

class HSIClasso(FeatureSelector):

    def __init__(self, dset):
        super().__init__(dset)
        hsic=HSICLasso()
        hsic.input(dset["X_train"], dset["y_train"][:, 1])
        hsic.classification(dset["X_train"].shape[1], B=20, discrete_x=True)

        self.max_k = len(hsic.A)
        self.importance = hsic.beta.ravel()
