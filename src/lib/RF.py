# import system modules
#
import os
import sys

# auxiliary imports
#
import numpy as np
import joblib

# SKLearn imports
#
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, classification_report

# import parameter file
#
import parameters as param
import Dataset as dc

class RF_MultiClassifier(MultiOutputClassifier):

    def __init__(self):
        rf = RandomForestClassifier(n_estimators=100, 
                                    max_depth=10)
        super().__init__(estimator=rf, n_jobs=-1)

        self.name = "RF"    

    def preprocess(self, X:np.ndarray, verbose=False) -> np.ndarray:
        X = dc.normalize(X, verbose=verbose)
        return dc.extract_features(X, feat_args=param.feat_methods, verbose=verbose)

    def fit(self, X:np.ndarray, y:np.ndarray) -> bool:
        
        try:
            super().fit(X,y)
            return True
        except Exception as e:
            print(e)
            return False
        
    def _postprocess(self, y: np.ndarray, verbose:bool=False) -> np.ndarray:
        
        output = np.zeros((len(y[0]), len(y)))

        for i, label in enumerate(y):
            for j, sample in enumerate(label):
                if sample[1] > param.threshold:
                    output[j][i] = 1
                else:
                    output[j][i] = 0

            if verbose: print(f"Postprocessed: {i+1}/{len(y)} lables")

        return output

    def predict(self, X: np.ndarray, verbose:bool=False) -> np.ndarray:

        try:
            preds = np.array(super().predict_proba(X))
            return self._postprocess(preds, verbose=verbose)
        except Exception as e:
            print(f"ERROR: {e}")
            return None
        
    def score(self, ref:np.ndarray, preds:np.ndarray, report:bool=False):
        
        if report:
            return classification_report(ref, preds, zero_division=0.0)
        else:
            return \
            [accuracy_score(ref[:, i], preds[:, i]) for i in range(ref.shape[1])]

        
    def save(self, fname:str) -> bool:

        try:    
            joblib.dump(self, fname)
            return True
        except Exception as e:
            print(e)
            return False
        
    def __str__(self) -> str:
        return super().n_layers
    
