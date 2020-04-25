import os

# Disable GPU Acceleration as Model will most likely not fit into memory. If you are running a Super-High-End GPU
# like the Quadro RTX 8000 you may enable GPU support
DISABLE_GPUS = True
if DISABLE_GPUS:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import yaml
import sys
import dill
import codecs
import pickle
import numpy as np
import pandas as pd
import tqdm
from collections import defaultdict
from sklearn.metrics import mean_squared_error
from keras.preprocessing.sequence import pad_sequences
from clint.textui import puts_err, puts


from keras.models import Model, load_model
from keras.layers import Input,Dense, Embedding, Reshape, Conv1D, MaxPooling1D, LSTM
from keras.layers import Dot, Multiply, Concatenate, Activation, Flatten, BatchNormalization, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils.vis_utils import model_to_dot

class CRSum:

    def __init__(self, M, N, verbose=True, configfile="config.yml"):

        self.M = M
        self.N = N
        self.labels = ["stm"+str(m+1) for m in reversed(range(M))] + ["st"] + ["stn"+str(m+1) for m in range(M)]
        self.padlen = 45
        self.verbose = verbose

        with open(configfile, "r") as infile:
            cfg = yaml.load(infile, Loader=yaml.SafeLoader)
        self.datadir = cfg["DataDir"]
        self.modeldir = cfg["ModelDir"]

        self.presentLangs = None

    def loadTrainingData(self):

        self.train_df = pd.read_pickle(os.path.join(self.datadir,"train_set.pkl"))
        self.presentLangs = set(list(train_df.Language.unique()) + list(test_df.Language.unique()))

    def loadTestData(self):

        self.test_df = pd.read_pickle(os.path.join(self.datadir,"test_set.pkl"))

    def loadEmbeddings(self):

        if self.presentLangs is None:
            puts_err("ERROR: Training data has to be loaded prior to the embeddings!")
            sys.exit()
