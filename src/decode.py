#!/usr/bin/env python

import os
import sys

import numpy as np
import joblib

import nedc_cmdl_parser as ncp

sys.path.append("/home/tuo72868/ece_8527_final/src/lib/")
import Dataset as ds
import parameters as param

HYP_NAME = "hyp_{}_{}.csv"

# define the help file and usage message:
#  since this is released software, we use an src directory
#
USAGE_FILE = "/home/tuo72868/ece_8527_final/src/lib/decode.usage"
HELP_FILE = "/home/tuo72868/ece_8527_final/src/lib/decode.help"

# define the program options:
#  note that you cannot separate them by spaces
#
ARG_ODIR = "--odir"
ARG_ABRV_ODIR = "-o"

ARG_MODEL = "--model"
ARG_ABRV_MODEL = "-m"

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

    # parse the command line
    #
    args = cmdl.parse_args()

    # check if the proper number of lists has been provided
    #
    if len(args.files) != NUM_ARGS:
        cmdl.print_usage('stdout')
        sys.exit(os.EX_SOFTWARE)

    fname = os.path.realpath(args.files[0].strip())

    model = joblib.load(args.model.strip())

    print("Retrieving data...")
    if fname.endswith(".list"):
        #X, y = ds.load_from_file(fname, verbose=True)
        X = ds._get_data(fname)
        X = model.preprocess(X, verbose=True)
    elif fname.endswith(".pkl"):
        X, y = ds.load_from_pkl(fname)

    print(model)

    print("Evaluating model...")
    preds = model.predict(X)

    print("Saving results...")
    odir = os.path.realpath(args.odir.strip())
    np.savetxt(odir, preds, delimiter=",", 
            header=",".join(param.label_names), comments='', fmt='%.0f')
    
    print("Done!")


if __name__ == "__main__":
    main(sys.argv)