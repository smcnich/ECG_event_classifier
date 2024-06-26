import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

from tensorflow.keras import models, layers
import pickle as pkl

sys.path.append("/home/tuo72868/ece_8527_final/src/util/")
from Dataset import Dataset

class CNN():

    def __init__(self, timesteps:int, num_channels:int, num_labels:int, *,
                 filter:int=16) -> None:

        self.timesteps = timesteps
        self.num_channels = num_channels

        inputs = [timesteps, num_channels]

        self.model = models.Sequential([
            layers.InputLayer(inputs),
            layers.Conv1D(filters=filter, kernel_size=2, activation='relu'),
            layers.MaxPooling1D(2),
            layers.Conv1D(filters=2*filter, kernel_size=2, activation='relu'),
            layers.MaxPooling1D(2),
            layers.Conv1D(filters=2*(2*filter), kernel_size=2, activation='relu'),
            layers.MaxPooling1D(2),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(2*(2*filter), activation='relu'),
            # Sigmoid for multi-label classification
            layers.Dense(num_labels, activation='sigmoid')
        ])

        self.model.compile(loss='binary_crossentropy', 
                           optimizer='adam', 
                           metrics=['accuracy'])
        
    def fit(self, train:Dataset, dev:Dataset,*, batch:int=32) -> bool:
        
        try:
            self.model.fit(train.X, train.y, 
                       epochs=10, batch_size=batch, 
                        validation_data=(dev.X, dev.y))
            return True
        except Exception as e:
            print(e)
            return False
        
    def evaluate(self, ds:Dataset) -> tuple:

        loss, accuracy = self.model.evaluate(ds.X, ds.y)

        return loss, accuracy
    
    def summary(self):
        self.model.summary()

    def save(self, fname:str) -> bool:

        try:
            with open(fname, 'wb') as fp:
                pkl.dump(self, fp, protocol=4)
            return True
        
        except Exception as e:
            print(e)
            return False
