import torch
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
from util import update_kwargdict

from torch.utils.data import Dataset

class Preprocessor(object):

    def __init__(self, scaler, test_split=0.1, val_split = 0.2, preimputation=True,
                 noise=True, **kwargs):
        """
        class that performs a priori noise injection, (normal/standard)ization,
        test/validation splitting and intitial preimputation. Output from the
        `transform' method can be used as input for `MaskedDataset' class.
        
        :param scaler: Scaler class from sklearn.preprocessing
        :param test_split: Percentage of dataset to use for testing
        :param val_split: Percentage of dataset to use for validation
        :param preimputation: Boolean whether to preimpute missing values
        :param noise: Boolean whether to inject a priori noise
        :param **kwargs: Keyword arguments for preimputation and noise
        """
        
        args = {
            "noise_args": {"distribution": "uniform",
                          "size": 0.5,
                          "before_scaler_fit": True,
                          "return": True},
            "preimpute_args": {"fn":"zero"}
        }
        update_kwargdict(args, kwargs)
        
        self._make_testval_params(test_split, val_split)

        self._make_scaler()

        self.preimputation = preimputation
        self.noise = noise
        if self.noise: self.noiseargs = args["noise_args"]
        if self.preimputation: self.preimputation_args = args["preimpute_args"]
        

    def fit(self, X, masks=None, noise_transform=True):
        """
        compute indices for test/validation splitting and parameters for scaling
        :param X: Full data numpy array with shape [n_samples, n_features] 
        :param mask: Mask(s) of missing value locations that should not be
                considered when scaling the data.
        :param noise_transform: Boolean whether to include noise injection before
                scaling the data (recommended).
        """       
        n = X.shape[0]
        p = X.shape[1]
        Xnan = np.copy(X)
        X_orig = np.copy(X)
        if len(masks) == 2:
            mask = mask[0]*mask[1]
        else:
            mask = masks
        
        # add noise, if requested before transform
        add_noise = noise_transform and self.noise
        if add_noise and self.noiseargs["before_scalar_fit"]:
            if self.noiseargs["distribution"]["type"].lower() == "uniform":
                level = self.noiseargs["distribution"]["size"]
                noise = np.random.uniform(-level,level,[n,p])
            elif self.noiseargs["distribution"]["type"].lower() in ["loggaussian","lognormal",
                                                                    "log_gaussian","log_normal"]:
                sigma = self.noiseargs["distribution"]["size"]
                noise = np.random.lognormal(mean=0.0,sigma=scale,[n,p])
            elif self.noiseargs["distribution"]["type"].lower() in ["gaussian","normal"]:
                scale = self.noiseargs["distribution"]["size"]
                noise = np.random.normal(loc=0.0,scale=scale,[n,p])
            self.noise_fit = noise
            Xnan = Xnan + noise
            
        Xnan[mask==0]=np.nan  
        npsize = np.arange(n)
        if self.test:
            test_n = int(n*self.test["split"])
            test_idx = np.random.choice(npsize, size=test_n, replace=False)
            self.test["indices"] = test_idx
            train_idx = np.delete(npsize, test_idx)
            
            #run scaler; train and validation use the same scaler settings
            self.test["scaler"].fit(Xnan[test_idx,:])
            self.train["scaler"].fit(Xnan[train_idx,:])
            if self.validation:
                val_n = int(n*self.validation["split"])
                val_idx = np.random.choice(train_idx, size=val_n, replace=False)
                self.validation["indices"] = val_idx
                train_idx = np.delete(train_idx, val_idx)
            self.train["indices"] = train_idx
        elif self.validation:
            self.train["scaler"].fit(Xnan)
            val_n = int(n*self.validation["split"])
            val_idx = np.random.choice(npsize, size=val_n, replace=False)
            train_idx = np.delete(train_idx, val_idx)
            self.validation["indices"] = val_idx
            self.train["indices"] = train_idx
        else:
            self.train["scaler"].fit(Xnan)
            self.train["indices"] = npsize
        self.original["scaler"].fit(X_orig)
        
        
    def transform(self, X, mask=None, preimpute_transform=True, noise_transform=True):
        """
        transform and return preprocessed data.
        
        Parameters
        ----------
        X : {array-like}, shape [n_samples, n_features]
            The full data used for train/test/validation to be scaled.
        mask: tuple or list of {array-like}, shape [n_samples, n_features]
            Masks of missing value locations. Either one mask for true missing,
            or two masks for both true and spiked-in missing.
        """
        returns=dict()
        n = X.shape[0]
        p = X.shape[1]
        X_ = np.copy(X)
        
        if mask is not None:
            # is it a tuple with true and spiked mask?
            mask_spike_true = True if len(mask)==2 else False
        
        # add noise, if requested before transform
        add_noise = noise_transform and self.noise
        if add_noise and self.noiseargs["before_scaler_fit"]:
            noise = self.noise_fit
            X_ = X_ + noise
        elif add_noise and not self.noiseargs["before_scaler_fit"]:
            if self.noiseargs["distribution"]["type"].lower() == "uniform":
                level = self.noiseargs["distribution"]["size"]
                noise = np.random.uniform(-level,level,[n,p])
            elif self.noiseargs["distribution"]["type"].lower() in ["loggaussian","lognormal",
                                                                    "log_gaussian","log_normal"]:
                sigma = self.noiseargs["distribution"]["size"]
                noise = np.random.lognormal(mean=0.0,sigma=scale,[n,p])
            elif self.noiseargs["distribution"]["type"].lower() in ["gaussian","normal"]:
                scale = self.noiseargs["distribution"]["size"]
                noise = np.random.normal(loc=0.0,scale=scale,[n,p])
            X_ = X_ + noise
        
        X_orig self.original["scaler"].transform(X_orig)
        # train/test/val split and scale
        X_train = np.copy(X_)[self.train["indices"]]
        X_train = self.train["scaler"].transform(X_train)
        returns["train"]=dict()
        if mask is not None:
            if mask_spike_true:
                masks_train = [m[self.train["indices"]] for m in mask]
                mask_train_preimpute = masks_train[0]*masks_train[1]
            else:
                masks_train = mask[self.train["indices"]]
                mask_train_preimpute = masks_train
            if preimpute_transform:
                _imp = preimpute(X_train, mask_train_preimpute, fn=self.preimputation_args["fn"])
                returns["train"]["data_preimp"] = _imp
                if mask_spike_true:
                    X_train[masks_train[0]==0] = np.copy(_imp[masks_train[0]==0])                
            returns["train"]["masks"] = masks_train
        returns["train"]["data"] = X_train
        returns["original"]["data"]

        if self.test:
            X_test = np.copy(X_)[self.test["indices"]]
            X_test = self.test["scaler"].transform(X_test) 
            returns["test"]=dict()
            if mask is not None:
                if mask_spike_true:
                    masks_test = mask[0][self.test["indices"]] # there should be no spiked for testing
                else:
                    masks_test = masks_test[self.test["indices"]]
                if preimpute_transform:
                    _imp = preimpute(X_test, mask_test_preimpute, fn=self.preimputation_args["fn"])
                    returns["validation"]["data_preimp"] = _imp
                    if mask_spike_true:
                        X_test[masks_test[0]==0] = np.copy(_imp[masks_test[0]==0])
                returns["test"]["masks"] = masks_test
                returns["test"]["data"] = X_test

                
        if self.validation:
            X_val = np.copy(X_)[self.validation["indices"]]
            X_val = self.train["scaler"].transform(X_val)
            returns["validation"]=dict()
            if mask is not None:
                if mask_spike_true:
                    masks_val = [m[self.validation["indices"]] for m in mask]
                    mask_val_preimpute = masks_val[0]*masks_val[1]
                else:
                    masks_val = mask[self.validation["indices"]]
                    mask_val_preimpute = masks_val
                if preimpute_transform:
                    _imp = preimpute(X_val, mask_val_preimpute, fn=self.preimputation_args["fn"])
                    returns["validation"]["data_preimp"] = _imp
                    if mask_spike_true:
                        X_val[masks_val[0]==0] = np.copy(_imp[masks_val[0]==0])
                returns["validation"]["masks"] = masks_val
                returns["validation"]["data"] = X_val
        
        return returns
    
    def _make_testval_params(self, test_split, val_split):
        """
        Splits the data into a train/test/validation set and stores in self
        """
        if test_split>0.0:
            self.test ={"split": test_split}
            if val_split>0.0:
                self.validation={"split":val_split}
                self.train = {"split": 1.0-(val_split+test_split)}
            else:
                self.validation=False
                self.train = {"split": 1.0-test_split}
        elif val_split>0.0:
            self.test =False
            self.validation={"split":val_split}
            self.train = {"split": 1.0-val_split}
        else:
            self.test =False
            self.validation=False
            self.train = {"split": 1.0}
            
    def _make_scaler(self, scaler=None):
        # Create scaler
        if scaler.lower() in ["standardscaler","standard","standardization"]:
            self.train["scaler"] = StandardScaler()
            if self.test: self.test["scaler"] = StandardScaler()
        elif scaler.lower() in ["minmaxscaler","minmax","normalization"]:
            self.train["scaler"] = MinMaxScaler()
            if self.test: self.test["scaler"] = MinMaxScaler()
        elif scaler.lower() in ["quantiletransformer","quantile"]:
            self.train["scaler"] = QuantileTransformer()
            if self.test: self.test["scaler"] = QuantileTransformer()
        else:
            raise ValueError("No valid scaler given")
        if self.validation: self.validation["scaler"] = self.train["scaler"]

        
    
def preimpute(X, mask,fn="zero"):
    X_imp = np.copy(X)
    X_imp[mask==0] = 0.
    return X_imp