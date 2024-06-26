import os
import sys

from util import CNN as cnn
import pickle as pkl

sys.path.append("/home/tuo72868/ece_8527_final/src/util/")
import Dataset as ds

# define constants
MODEL_DIR = "/home/tuo72868/ece_8527_final/cnn/model"
MODEL_FNAME = "cnn_{}.tf"

PKL_DIR = "/home/tuo72868/ece_8527_final/cnn/data"

# Define parameters
num_samples = 1000
num_channels = 8
samp_freq = 300
seq_len = 2200
label_names = ['1dAVb', 'RBBB', 'LBBB', 'SB', 'AF', 'ST']

def load_from_file(fname:str,*, verbose:bool=False) -> ds.Dataset:

    # create a dataset object and retrieve the data from the list file
    if verbose: print("Loading data into memory...")
    dst = ds.Dataset(num_channels=num_channels, 
                    samp_freq=samp_freq, 
                    label_names=label_names,
                    seq_len=seq_len)
    dst.get_data(fname)

    # find the annotation csv file and retrieve the annotations
    fname_csv = os.path.splitext(fname)[0] + '.csv'
    fname_csv = os.path.realpath(fname_csv)
    dst.get_ann(fname_csv)

    # serialize the data for ease of access
    if verbose: print("Pickling data...")
    fname_pkl = os.path.splitext(fname)[0] + ".pkl"
    fname_pkl = os.path.basename(fname_pkl)
    fname_pkl = os.path.join(PKL_DIR, fname_pkl)
    with open(fname_pkl, 'wb') as fp:
        pkl.dump(dst, fp, protocol=4)

    # return the dataset object
    return dst
#
# end of method

def load_from_pkl(fname:str) -> ds.Dataset:
    
    # open a serialized file and retrieve the object stored inside it
    with open(fname, 'rb') as file:
        loaded_object = pkl.load(file)

    # return the loaded object
    return loaded_object
#
# end of method

def main(argv):

    # load a list file or pickle file
    print("Loading data into memory...")
    if argv[1].strip().endswith(".list"):
        train = load_from_file(os.path.realpath(argv[1].strip()), verbose=True)
        train.normalize()
    elif argv[1].strip().endswith(".pkl"):
        train = load_from_pkl(os.path.realpath(argv[1].strip()))

    # load a list file or pickle file
    if argv[2].strip().endswith(".list"):
        dev = load_from_file(os.path.realpath(argv[2].strip()), verbose=True)
        train.normalize()
    elif argv[2].strip().endswith(".pkl"):
        dev = load_from_pkl(os.path.realpath(argv[2].strip()))

    model = cnn.CNN(timesteps=seq_len, num_channels=num_channels,
                    num_labels=len(label_names), filter=32)

    model.summary()

    print("Fitting a model...")
    model.fit(train=train, dev=dev, batch=16)

    print("Evaluating the model...")
    _, acc = model.evaluate(dev)

    print(f"Model accuracy: {acc*100:.2f}%")

    if acc > 0.75:

        fname = os.path.join(MODEL_DIR, MODEL_FNAME.format(round(acc*100)))

        model.save(os.path.realpath(fname))


if __name__ == "__main__":
    main(sys.argv)