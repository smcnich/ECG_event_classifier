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
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
from tensorflow.keras import models, layers
from sklearn.metrics import accuracy_score

# import parameter file
#
import parameters as param
import Dataset as dc

class CNN_MultiClassifier(models.Sequential):

    def __init__(self):

        inputs = [param.seq_len, param.num_channels]

        super().__init__()       
        super().add(layers.InputLayer(inputs))
        super().add(layers.Conv1D(filters=param.layer_size[0], kernel_size=2, activation='relu'))
        super().add(layers.MaxPooling1D(2))
        super().add(layers.Conv1D(filters=param.layer_size[1], kernel_size=2, activation='relu'))
        super().add(layers.MaxPooling1D(2))
        super().add(layers.Conv1D(filters=param.layer_size[2], kernel_size=2, activation='relu'))
        super().add(layers.MaxPooling1D(2))
        super().add(layers.Flatten())
        super().add(layers.Dropout(0.5))
        super().add(layers.Dense(param.layer_size[2], activation='relu'))
        super().add(layers.Dense(param.label_count, activation='sigmoid'))
        
        super().compile(loss='binary_crossentropy', 
                      optimizer='adam', 
                      metrics=['accuracy'])
   

    def preprocess(self, X:np.ndarray, verbose=False) -> np.ndarray:
       
        return dc.normalize(X, verbose=verbose)

    def fit(self, X:np.ndarray, y:np.ndarray, validation_data:tuple) -> bool:

        try:
            super().fit(X, y,
                        validation_data=validation_data,
                        epochs=param.epochs,
                        batch_size=param.batch_size)
            return True
        except Exception as e:
            print(f"ERROR: {e}")
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
            return super().predict(X)
        except Exception as e:
            print(f"ERROR: {e}")
            return None

        
    def score(self, ref:np.ndarray, preds:np.ndarray) -> list:
        return \
        [accuracy_score(ref[:, i], preds[:, i]) for i in range(ref.shape[1])]

        
    def save(self, fname:str) -> bool:

        try:    
            joblib.dump(self, fname)
            return True
        except Exception as e:
            print(e)
            return False
    
