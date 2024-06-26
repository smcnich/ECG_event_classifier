#!/usr/bin/env python

import os
import sys

import pickle as pkl

import nedc_cmdl_parser as ncp

sys.path.append("/home/tuo72868/ece_8527_final/src/lib/")
import Dataset as ds
import parameters as param


# define the help file and usage message:
#  since this is released software, we use an src directory
#
USAGE_FILE = "/home/tuo72868/ece_8527_final/src/lib/dataload.usage"
HELP_FILE = "/home/tuo72868/ece_8527_final/src/lib/dataload.help"

MODEL_FNAME = "{}.pkl"

# define the program options:
#  note that you cannot separate them by spaces
#
ARG_ODIR = "--odir"
ARG_ABRV_ODIR = "-o"

ARG_NORM = "--normalize"
ARG_ABRV_NORM = "-n"

ARG_FEAT = "--features"
ARG_ABRV_FEAT = "-f"

ARG_PCA = "--pca"
ARG_ABRV_PCA = "-p"

NUM_ARGS = 1

def main(argv):

    # create a command line parser
    #
    cmdl = ncp.Cmdl(USAGE_FILE, HELP_FILE)

    # define the command line arguments
    #
    cmdl.add_argument("files", type = str, nargs = '*')
    cmdl.add_argument(ARG_ABRV_ODIR, ARG_ODIR, type = str)
    cmdl.add_argument(ARG_ABRV_NORM, ARG_NORM, action="store_true")
    cmdl.add_argument(ARG_ABRV_FEAT, ARG_FEAT, action="store_true")
    cmdl.add_argument(ARG_ABRV_PCA, ARG_PCA, action="store_true")

    # parse the command line
    #
    args = cmdl.parse_args()

    # check if the proper number of lists has been provided
    #
    if not len(args.files) >= NUM_ARGS:
        cmdl.print_usage('stdout')
        sys.exit(os.EX_SOFTWARE)

    for fname in args.files:

        fname = fname.strip()

        if fname.endswith(".list"):

            print(f"====={os.path.basename(fname)}=====")

            fname = os.path.realpath(fname.strip())

            print(" Retrieving data...")
            #X, y = ds.load_from_file(fname)
            X = ds._get_data(fname)

            print(" Normalizing and extracting features...")
            if args.features and args.pca:
                print("Cannot extract features and perform PCA")
                return 1

            if args.normalize:
                X = ds.normalize(X, verbose=True)

            if args.features:
                X = ds.extract_features(X, param.feat_methods, verbose=True)

            if args.pca:
                X = ds.pca(X, verbose=True)

            # serialize the data for ease of access
            print(" Pickling data...")
            fname_pkl = os.path.basename(fname).split('.')[0]
            fname_pkl = os.path.join(os.path.realpath(args.odir.strip()), 
                                    fname_pkl + ".pkl")
            
            with open(fname_pkl, 'wb') as fp:
                pkl.dump((X), fp, protocol=4)

        else:
            print(f"Cannot process file {fname}")
            return 1
        
        return 0

if __name__ == "__main__":
    main(sys.argv)