#!/usr/local/bin/python3

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
import pickle

import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import mean_squared_error
from tensorflow.keras.preprocessing.sequence import pad_sequences
from clint.textui import puts_err, puts, prompt, colored

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Embedding, Reshape, Conv1D, MaxPooling1D, LSTM
from tensorflow.keras.layers import Dot, Multiply, Concatenate, Activation, Flatten, BatchNormalization, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint


class CRSum:

    def __init__(self, embedding_model, preprocessor, M=5, N=5, verbose=True, configfile="config.yml"):

        self.M = M
        self.N = N
        self.labels = ["stm" + str(m + 1) for m in reversed(range(M))] + ["st"] + ["stn" + str(n + 1) for n in range(N)]
        self.verbose = verbose

        with open(configfile, "r") as infile:
            cfg = yaml.load(infile, Loader=yaml.SafeLoader)
        self.datadir = cfg["Supervised"]["DataDir"]
        self.modeldir = cfg["Supervised"]["ModelDir"]
        self.pad_len = cfg["Supervised"]["PadLength"]
        self.embeddingdir = cfg["General"]["MatrixDirectory"]
        #self.embeddingdir = embedding_model.get_matrix_directory()
        self.prepro = preprocessor
        self.lang_prepos = None

        # Initialize Variables for later use
        self.present_langs = None
        self.master_matrix = np.zeros((0, 300))
        self.lang_wordmappers = {}
        self.model = None
        self.train_df = None
        self.test_df = None
        self.embsLoaded = False
        self.scaler = None

    def loadTrainingData(self):
        if self.verbose:
            puts("Loading training data...")
        self.train_df = pd.read_pickle(os.path.join(self.datadir, "train_set.pkl"))
        self.present_langs = list(self.train_df.Language.unique())
        if self.verbose:
            puts("Done.")

    def loadTestData(self):

        if self.verbose:
            puts("Loading test data...")

        if self.scaler is None:
            self._load_scaler()

        self.test_df = pd.read_pickle(os.path.join(self.datadir, "test_set.pkl"))
        self.test_df[['len', 'df', 'tf']] = self.scaler.transform(self.test_df[['len', 'df', 'tf']])
        self.present_langs = list(self.test_df.Language.unique())
        if self.verbose:
            puts("Done.")

    def loadEmbeddings(self):

        if os.path.isfile(os.path.join(self.datadir, "mastermapper.dill")) and os.path.isfile(os.path.join(self.datadir, "mastermatrix.pkl")):

            if self.verbose:
                puts("Loading prebuilt embedding matrices.")

            with open(os.path.join(self.datadir, "mastermapper.dill"), 'rb') as file:
                self.lang_wordmappers = dill.load(file)

            with open(os.path.join(self.datadir, "mastermatrix.pkl"), 'rb') as file:
                self.master_matrix = pickle.load(file)

            self.present_langs = self.lang_wordmappers.keys()

            if self.verbose:
                puts("Done.")
        else:

            if self.verbose:
                puts("No prebuilt matrices found. Initializing master matrix.")

            if self.present_langs is None:
                puts_err("ERROR: Cannot initialize master matrix without data! Please load training data first")
                sys.exit()

            lang_matrices = {}

            if self.verbose:
                puts("Loading embeddings and mappers of present languages:")

            for l in self.present_langs:

                if self.verbose:
                    puts(l)

                with open(os.path.join(self.embeddingdir, "wordmapper_" + l + ".dill"), 'rb') as file:
                    self.lang_wordmappers[l] = dill.load(file)

                with open(os.path.join(self.embeddingdir, "emb_matrix_" + l + ".pck"), 'rb') as file:
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

            if self.verbose:
                puts("Done.")
            del lang_matrices

            if prompt.yn("Successfully built master matrix. Do you want to save the matrix for future use?"):
                with open(os.path.join(self.datadir, "mastermapper.dill"), 'wb') as file:
                    dill.dump(self.lang_wordmappers, file)

                with open(os.path.join(self.datadir, "mastermatrix.pkl"), 'wb') as file:
                    pickle.dump(self.master_matrix, file, protocol=4)

        self.embsLoaded = True

    def _load_scaler(self):
        with open(os.path.join(self.datadir,"scaler.pkl"), 'rb') as file:
            self.scaler = pickle.load(file)

    def wordToIndex(self, data):
        rowlang = data.Language
        #if len(data.iloc[0]) > 0:
        return [self.lang_wordmappers[rowlang][word] for word in data.iloc[0]]
        #else:
        #    return []

    def process_text(self, data):
        return pad_sequences(data.apply(self.wordToIndex, axis=1), maxlen=self.pad_len, padding='post',
                             truncating='post')

    def get_inputs(self, df):
        inputs = {}
        for label in self.labels:
            if self.verbose:
                puts("Processing "+label)
            inputs[label] = self.process_text(df[[label, "Language"]])
        inputs['sf'] = np.array(df[['len', 'pos', 'tf', 'df']])
        return inputs

    def makeModel(self, sentence_dim=50, dense_dim=100, dense_depth=3, dropout=0.1):

        embedding_layer = Embedding(input_dim=self.master_matrix.shape[0], output_dim=self.master_matrix.shape[1],
                                    weights=[self.master_matrix], trainable=False)

        def get_att_sentence(name, conv_layer, st, pdl):
            inp = Input(shape=(pdl,), name=name)
            emb = embedding_layer(inp)
            conv = conv_layer(emb)
            weight = Activation('softmax')(Dot(axes=2, normalize=True)([conv, st]))
            att = Multiply()([weight, conv])
            mp = MaxPooling1D(pdl - 1)(att)
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
            inp, sen = get_att_sentence(name, conv_layer, st, self.pad_len)
            inputs.append(inp)
            pc.append(sen)

        fc = []
        for i in range(1, self.N + 1):
            name = 'stn' + str(i)
            inp, sen = get_att_sentence(name, conv_layer, st, self.pad_len)
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
        output = Dense(1, activation='tanh')(main)

        model = Model(inputs=inputs, outputs=output)
        model.compile(optimizer='adam', loss='mse')
        return model

    def train(self, epochs=20, batch_size=1024):

        if not os.path.isdir(self.modeldir):
            puts("The model directory you've configured does not yet exist.")
            if prompt.yn("Do you want to create the directory " + colored.green(self.modeldir) + " now?"):
                os.mkdir(self.modeldir)
            else:
                puts_err("Please define an existing ModelDir in config.yaml. Bye!")
                sys.exit()

        if self.train_df is None:
            self.loadTrainingData()

        if not self.embsLoaded:
            if self.verbose:
                self.loadEmbeddings()

        if self.verbose:
            puts("Preparing training data...")

        X_train = self.get_inputs(self.train_df)
        Y_train = self.train_df.cosine_sim.values.reshape(-1, 1)

        if (self.verbose):
            puts("Done.")
            puts("Building model...")
        self.model = self.makeModel()
        if self.verbose:
            puts("Done.")
        checkpoint_path = os.path.join(self.modeldir, "epoch{epoch:03d}-{loss:.8f}.h5")
        callbacks = [ModelCheckpoint(checkpoint_path, verbose=1, monitor='loss', save_best_only=True, mode='auto')]
        self.model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, callbacks=callbacks)
        #"""

    def eval(self):
        if self.model is None:
            puts_err("Train model or load weights before evaluation!")
            sys.exit()

        if self.test_df is None:
            self.loadTestData()

        if not self.embsLoaded:
            self.loadEmbeddings()

        if self.verbose:
            puts("Preparing test data...")

        X_test = self.get_inputs(self.test_df)

        if (self.verbose):
            puts("Done.")
            puts("Computing loss...")

        self.test_df['pred_rank'] = self.model.predict(X_test, batch_size=2048)
        self.test_loss = mean_squared_error(self.test_df.cosine_sim, self.test_df.pred_rank)
        puts("Mean Squared Error: " + str(self.test_loss))

    def loadWeights(self, filename):
        self.model = load_model(os.path.join(self.modeldir, filename))

    def predict(self, data):
        if not self.embsLoaded:
            self.loadEmbeddings()
        if self.scaler is None:
            self._load_scaler()

        data[['len', 'df', 'tf']] = self.scaler.transform(data[['len', 'df', 'tf']])

        X_data = self.get_inputs(data)
        data['pred_rank'] = self.model.predict(X_data, batch_size=2048)
        return data

    def summarize(self, text, language, sum_len):

        if self.model is None:
            puts_err("No weights loaded. Please load weigths first by calling method loadWeights()")
            sys.exit()

        if not self.embsLoaded:
            self.loadEmbeddings()

        if language not in self.present_langs:
            puts_err("Unsupported language. Currently supported: "+" ".join(self.present_langs))
            sys.exit()

        if self.lang_prepos is None:
            self.lang_prepos = {}
            for l in self.present_langs:
                self.lang_prepos[l] = self.prepro(l, self.M, self.N)

        pred_df, tok_text = self.lang_prepos[language].preprocess(text)
        # If length is smaller than 1, compute the number of sentences the fraction corresponds to
        if sum_len < 1:
            sum_len = round(sum_len * len(tok_text))
        else:
            sum_len = int(sum_len)

        pred = self.predict(pred_df)
        highest_sim = pred.nlargest(sum_len, columns=['pred_rank']).index
        highest_sim = [int(i) for i in highest_sim]
        highest_sim.sort()
        summary = [tok_text[int(i)] for i in highest_sim]
        return " ".join(summary)


from Preprocessors import CRSumPreprocessor
from clint.textui import puts, prompt, colored

if __name__ == "__main__":
    textfile = input("Enter the path to the text file you want to summarize: ")
    with open(textfile, 'r') as file:
        longtext = file.read().replace('\n', ' ')
    l = input("Which language is the text? ")
    sentcount = float(input("How long should the summary be?\nNumber of sentences or fraction of total: "))
    csm = CRSum(embedding_model=None, preprocessor=CRSumPreprocessor, M=5, N=5)
    csm.loadWeights("epoch020-0.00124268.h5")
    summary = csm.summarize(longtext, l, 0.5)
    puts("Summary:")
    puts(colored.blue(summary))
    puts("Original text:")
    puts(colored.magenta(longtext))
