#!/usr/bin/env python

import os
import sys

import nedc_cmdl_parser as ncp

sys.path.append("/home/tuo72868/ece_8527_final/src/lib/")
import Dataset as ds
import parameters as param


# define the help file and usage message:
#  since this is released software, we use an src directory
#
USAGE_FILE = "/home/tuo72868/ece_8527_final/src/lib/decode.usage"
HELP_FILE = "/home/tuo72868/ece_8527_final/src/lib/decode.help"

MODEL_FNAME = "{}.pkl"

# define the program options:
#  note that you cannot separate them by spaces
#
ARG_ODIR = "--odir"
ARG_ABRV_ODIR = "-o"

ARG_MODEL = "--model"
ARG_ABRV_MODEL = "-m"

ARG_DEV = "--dev"
ARG_ABRV_DEV = "-d"

NUM_ARGS = 1

def main(argv):

    # create a command line parser
    #
    cmdl = ncp.Cmdl(USAGE_FILE, HELP_FILE)

    # define the command line arguments
    #
    cmdl.add_argument("files", type = str, nargs = '*')
    cmdl.add_argument(ARG_ABRV_ODIR, ARG_ODIR, type = str)
    cmdl.add_argument(ARG_ABRV_MODEL, ARG_MODEL, type = str)
    cmdl.add_argument(ARG_ABRV_DEV, ARG_DEV, type = str)

    # parse the command line
    #
    args = cmdl.parse_args()

    # check if the proper number of lists has been provided
    #
    if len(args.files) != NUM_ARGS:
        cmdl.print_usage('stdout')
        sys.exit(os.EX_SOFTWARE)

    files = []
    for file in args.files:
        files.append(os.path.realpath(file.strip()))

    model = param.models[args.model]()

    print("Retrieving data...")
    if files[0].endswith(".list"):
        X, y = ds.load_from_file(files[0], verbose=True)
        X = model.preprocess(X, verbose=True)
    elif files[0].endswith(".pkl"):
        X, y = ds.load_from_pkl(files[0])

    if args.dev:
        dev = os.path.realpath(args.dev.strip())
        if dev.endswith(".list"):
            X_dev, y_dev = ds.load_from_file(dev, verbose=True)
            X_dev = model.preprocess(X_dev, verbose=True)
        elif files[0].endswith(".pkl"):
            X_dev, y_dev = ds.load_from_pkl(dev)

    print("Fitting model...")
    if args.dev:
        model.fit(X, y, (X_dev, y_dev))
    else:
        model.fit(X, y)

    print("Evaluating model...")
    preds = model.predict(X)

    odir = os.path.realpath(args.odir.strip())
    odir = os.path.join(odir, MODEL_FNAME.format(model.name))
    model.save(odir)

    # Calculating accuracy for each label separately
    print(model.score(y, preds, report=True))

    
if __name__ == "__main__":
    main(sys.argv)