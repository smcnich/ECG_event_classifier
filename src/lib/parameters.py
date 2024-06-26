import sys
sys.path.append("/home/tuo72868/ece_8527_final/src/lib")

import numpy as np
from scipy.stats import kurtosis, skew
from scipy.signal import welch

from RF import RF_MultiClassifier
from MLP import MLP_MultiClassifier
from CNN import CNN_MultiClassifier

PKL_DIR = "/home/tuo72868/ece_8527_final/data/"
MODEL_DIR = "/home/tuo72868/ece_8527_final/model/"
HYP_DIR = "/home/tuo72868/ece_8527_final/hyp/"

# Define general parameters
num_samples = 1000
num_channels = 8
samp_freq = 300
seq_len = 2200
label_names = ['1dAVb', 'RBBB', 'LBBB', 'SB', 'AF', 'ST']
label_count = len(label_names)

layer_size =(18, 36, 72)
epochs = 10
batch_size = 16

models = {"RF" : RF_MultiClassifier,
          "MLP":MLP_MultiClassifier,
          "CNN":CNN_MultiClassifier}

threshold = 0.55

def total_power(X:np.ndarray) -> float:
    # calculate power spectral density
    _, psd = welch(X, fs=samp_freq)
    return np.sum(psd)

def mean_power(X:np.ndarray) -> float:
    # calculate power spectral density
    _, psd = welch(X, fs=samp_freq)
    return np.mean(psd)

feat_methods = [np.mean, np.std, skew, kurtosis, np.amax, np.amin, np.ptp, 
                total_power, mean_power]
n_features = 8