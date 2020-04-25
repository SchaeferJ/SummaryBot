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
from keras.layers import Input, Dense, Embedding, Reshape, Conv1D, MaxPooling1D, LSTM
from keras.layers import Dot, Multiply, Concatenate, Activation, Flatten, BatchNormalization, Dropout
from keras.callbacks import ModelCheckpoint

class CRSum:

    def __init__(self, M, N, verbose=True, configfile="config.yml"):

        self.M = M
        self.N = N
        self.labels = ["stm" + str(m + 1) for m in reversed(range(M))] + ["st"] + ["stn" + str(m + 1) for m in range(M)]
        self.verbose = verbose

        with open(configfile, "r") as infile:
            cfg = yaml.load(infile, Loader=yaml.SafeLoader)
        self.datadir = cfg["Supervised"]["DataDir"]
        self.modeldir = cfg["Supervised"]["ModelDir"]
        self.embeddingdir = cfg["General"]["MatrixDirectory"]

        # Initialize Variables for later use
        self.present_langs = None
        self.master_matrix = np.zeros((0, 300))
        self.lang_wordmappers = {}
        self.pad_len = None
        self.model = None

    def loadTrainingData(self):

        self.train_df = pd.read_pickle(os.path.join(self.datadir, "train_set.pkl"))
        self.present_langs = set(list(self.train_df.Language.unique()) + list(self.test_df.Language.unique()))
        self.pad_len = int(np.percentile(self.train_df.st.apply(len).values, 99.7) - 2)

    def loadTestData(self):

        self.test_df = pd.read_pickle(os.path.join(self.datadir, "test_set.pkl"))

    def loadEmbeddings(self):

        if self.present_langs is None:
            puts_err("ERROR: Training data has to be loaded prior to the embeddings!")
            sys.exit()

        lang_matrices = {}

        if self.verbose:
            puts("Loading embeddings and mappers of present languages:")

        for l in self.present_langs:

            if self.verbose:
                puts(l)

            with open(os.path.join(self.embeddingdir, "wordmapper_" + l + ".dill", 'rb')) as file:
                self.lang_wordmappers[l] = dill.load(file)

            with open(os.path.join(self.embeddingdir, "emb_matrix_" + l + ".pck", 'rb')) as file:
                lang_matrices[l] = pickle.load(file)

            if self.verbose:
                puts("Done.")
                puts("Concatenating matrices:")

        offset = 0
        for l in self.present_langs:
            if self.verbose:
                puts(l)
            self.master_matrix = np.concatenate((self.master_matrix, lang_matrices[l]), axis=0)
            for key in self.lang_wordmappers[l]:
                self.lang_wordmappers[l][key] += offset
            offset += len(lang_matrices[l])

        del lang_matrices

    def wordToIndex(self, data):
        rowlang = data.Language
        if len(data.iloc[0]) > 0:
            return [self.lang_wordmappers[rowlang][word] for word in data.iloc[0]]
        else:
            return []

    def process_text(self, data):
        return pad_sequences(data.apply(self.wordToIndex, axis=1), maxlen=self.pad_len, padding='post',
                             truncating='post')

    def get_inputs(self, df):
        inputs = {}
        for label in self.labels:
            inputs[label] = self.process_text(df[[label, "Language"]])
        inputs['sf'] = np.array(df[['len', 'pos', 'tf', 'df']])
        return inputs

    def makeModel(self, sentence_dim=50, dense_dim=100, dense_depth=3, dropout=0.1):

        embedding_layer = Embedding(input_dim=self.master_matrix.shape[0], output_dim=self.master_matrix.shape[1],
                                    weights=[self.master_matrix], trainable=False)

        def get_att_sentence(name, conv_layer, st):
            inp = Input(shape=(self.pad_len,), name=name)
            emb = embedding_layer(inp)
            conv = conv_layer(emb)
            weight = Activation('softmax')(Dot(axes=2, normalize=True)([conv, st]))
            att = Multiply()([weight, conv])
            mp = MaxPooling1D(self.pad_len - 1)(att)
            return inp, mp

        inputs = []
        # surface features
        sf = Input(shape=(4,), name='sf')
        inputs.append(sf)
        # current sentence modeling
        inp_st = Input(shape=(self.pad_len,), name='st')
        emb_st = embedding_layer(inp_st)
        conv_layer = Conv1D(sentence_dim, 2, activation='tanh')
        conv_st = conv_layer(emb_st)
        st = MaxPooling1D(self.pad_len - 1)(conv_st)
        st_flatten = Flatten()(st)
        inputs.append(inp_st)

        # other sentences modeling
        pc = []
        for i in reversed(range(1, self.M + 1)):
            name = 'stm' + str(i)
            inp, sen = get_att_sentence(name, conv_layer, st)
            inputs.append(inp)
            pc.append(sen)

        fc = []
        for i in range(1, self.N + 1):
            name = 'stn' + str(i)
            inp, sen = get_att_sentence(name, conv_layer, st)
            inputs.append(inp)
            fc.append(sen)

        # context relation modeling
        lstm_layer = LSTM(sentence_dim, return_sequences=True, recurrent_dropout=dropout)

        h_pc, h_fc = [], []
        for sen in pc:
            h_pc.append(lstm_layer(sen))
        h_st = lstm_layer(st)
        for sen in fc:
            h_fc.append(lstm_layer(sen))

        h_pc = Concatenate(axis=1)(h_pc)
        w_pc = Activation('softmax')(Dot(axes=2, normalize=True)([h_pc, h_st]))
        att_pc = Multiply()([w_pc, h_pc])
        mp_pc = MaxPooling1D(self.M)(att_pc)
        cos_pc = Flatten()(Dot(axes=2, normalize=True)([mp_pc, st]))

        h_fc = Concatenate(axis=1)(h_fc)
        w_fc = Activation('softmax')(Dot(axes=2, normalize=True)([h_fc, h_st]))
        att_fc = Multiply()([w_fc, h_fc])
        mp_fc = MaxPooling1D(self.N)(att_fc)
        cos_fc = Flatten()(Dot(axes=2, normalize=True)([mp_fc, st]))

        main = Concatenate()([cos_pc, cos_fc, st_flatten, sf])

        main = BatchNormalization()(main)
        for _ in range(dense_depth):
            main = Dropout(dropout)(Dense(dense_dim, activation='relu')(main))
        output = Dense(1)(main)

        model = Model(inputs=inputs, outputs=output)
        model.compile(optimizer='adam', loss='mse')
        return model

    def train(self, epochs=20, batch_size=1024):
        if self.verbose:
            puts("Preparing training data...")
        X_train = self.get_inputs(self.train_df)
        Y_train = self.train_df.cosine_sim.values.reshape(-1, 1)

        if(self.verbose):
            puts("Done.")
            puts("Building model...")
        self.model = self.makeModel()
        if self.verbose:
            puts("Done.")
        checkpoint_path = os.path.join(self.modeldir, "epoch{epoch:03d}-{loss:.8f}.h5")
        callbacks = [ModelCheckpoint(checkpoint_path, verbose=1, monitor='loss', save_best_only=True, mode='auto')]
        self.model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, callbacks=callbacks)

    def eval(self):
        if self.verbose:
            puts("Preparing test data...")
        X_train = self.get_inputs(self.train_df)
        Y_train = self.train_df.cosine_sim.values.reshape(-1, 1)

        if(self.verbose):
            puts("Done.")