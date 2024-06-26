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
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# import parameter file
#
import parameters as param
import Dataset as dc

class MLP_MultiClassifier(MLPClassifier):

    def __init__(self):
        super().__init__(hidden_layer_sizes=(100,50,25,10),
                          activation='relu',
                          solver='adam',
                          max_iter=1000,
                          verbose=True)

        self.name = "MLP"    

    def preprocess(self, X:np.ndarray, verbose=False) -> np.ndarray:
        return dc.normalize(X, verbose=verbose)

    def fit(self, X:np.ndarray, y:np.ndarray) -> bool:
        
        try:
            super().fit(X,y)
            return True
        except Exception as e:
            print(f"ERROR: {e}")
            return False
        
    def _postprocess(self, y: np.ndarray, verbose:bool=False) -> np.ndarray:
        
        y[y < param.threshold] = 0
        y[y >= param.threshold] = 1

        return y

    def predict(self, X: np.ndarray, verbose:bool=False) -> np.ndarray:

        try:
            preds = np.array(super().predict_proba(X))
            return self._postprocess(preds, verbose=verbose)
            #return np.array(super().predict(X))
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
        return str(self.n_outputs_)
    
