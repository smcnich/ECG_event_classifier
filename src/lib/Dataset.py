import os

import numpy as np
import pickle as pkl
from sklearn.preprocessing import robust_scale
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import parameters as param


def _get_sig(fname:str) -> np.ndarray:

    data = np.fromfile(fname, dtype=np.int16)
    data = data.reshape(-1, param.num_channels)

    return data

def _get_label(labels:str) -> int:
    labels = labels.strip().split(',')
    try:
        idx = labels.index("1") + 1
    except ValueError:
        idx = 0
    return idx

def _get_ann(fname:str) -> np.ndarray:
    with open(fname, "r") as fp:

        anns = fp.readlines()[1:]

        y = np.zeros((len(anns), len(param.label_names)))
        for i, labels in enumerate(anns):
            y[i] = np.array(labels.strip().split(","))

    return y

def _get_data(fname:str) -> np.ndarray:
    with open(fname, "r") as fp:
        
        files = fp.readlines()

        X = np.zeros((len(files), param.seq_len, param.num_channels))

        for i, dat in enumerate(files):
            sig = _get_sig(os.path.realpath(dat).strip())

            X[i] = sig

    return X.reshape((len(X), -1))

def pca(X:np.ndarray, verbose:bool=False) -> np.ndarray:
    

    X = StandardScaler().fit_transform(X)

    # Apply PCA along the channel axis
    pca = PCA(n_components=param.n_features)
    return pca.fit_transform(X)


def normalize(X: np.ndarray, verbose=False) -> np.ndarray:

    norm_X = np.zeros(X.shape)
    for i, sample in enumerate(X):
        norm_X[i] = robust_scale(sample)

        if verbose: print(f"Normalized: {i+1}/{len(X)}")

    return np.array(norm_X)

def extract_features(X: np.ndarray, feat_args:list, verbose: bool = False):

    feats = np.zeros((len(X), len(feat_args)))
    for i, sample in enumerate(X):
        for j, feat in enumerate(feat_args):

            try:
                feats[i][j] = feat(sample)
            except ValueError:
                feats[i][j] = feat(sample.flatten())

            if np.isnan(feats[i]).any():
                # Find NaN positions in the features
                nan_positions = np.isnan(feats[i])

                # Iterate over each NaN position
                for pos, is_nan in enumerate(nan_positions):
                    if is_nan:
                        # Find the median of the column preceding the NaN position
                        preceding_values = feats[i, :pos]
                        median = np.median(preceding_values)
                        
                        # Replace NaN value with the median
                        feats[i, pos] = median


        if verbose: print(f"Features extracted: {i+1}/{len(X)}")

    return feats


def load_from_file(fname:str,*, verbose:bool=False) -> tuple:

    # create a dataset object and retrieve the data from the list file
    if verbose: print("Loading data into memory...")

    X = _get_data(fname)

    # find the annotation csv file and retrieve the annotations
    fname_csv = os.path.splitext(fname)[0] + '.csv'
    fname_csv = os.path.realpath(fname_csv)
    y = _get_ann(fname_csv)

    # return the dataset object
    return (X, y)
#
# end of method

def load_from_pkl(fname:str):
    
    # open a serialized file and retrieve the object stored inside it
    with open(fname, 'rb') as file:
        data = pkl.load(file)

    X = data[0]
    y = data[1]

    return (X, y)
#
# end of method

def pickle_data(X:np.ndarray, y:np.ndarray, fname:str) -> bool:

    try:
        with open(fname, 'wb') as fp:
            pkl.dump((X, y), fp, protocol=4)

        return True

    except Exception as e:
        print(e)
        return False