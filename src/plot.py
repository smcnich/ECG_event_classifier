#!/usr/bin/env python
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl

sys.path.append("/home/tuo72868/ece_8527_final/src/lib/")
import Dataset as ds
import parameters as param

# Create a time axis (assuming sampling rate of 1000 Hz)
duration_per_ecg = param.seq_len / param.samp_freq

time = np.linspace(0, duration_per_ecg, param.seq_len, endpoint=False)

X = ds._get_sig(os.path.realpath(sys.argv[1].strip()))

#X = ds.normalize(np.array([X]))[0]

plt.figure(figsize=(10, 6))
for i in range(param.num_channels):
    plt.subplot(param.num_channels, 1, i+1)
    plt.plot(time, X[:, i])
    plt.grid(True)

#plt.tight_layout()  # Adjust subplot layout to prevent overlap
plt.show()