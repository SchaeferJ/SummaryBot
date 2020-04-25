#!/usr/local/bin/python3

import nltk
import string
from nltk.corpus import stopwords
from clint.textui import puts_err

import nltk
import numpy as np
import string
import pandas as pd
from collections import Counter, defaultdict
from nltk.corpus import stopwords
from clint.textui import puts_err
from nltk.tokenize import sent_tokenize


class StandardPreprocessor:
    """"Standard Preprocessor for sentence embedding"""

    def __init__(self, language, remove_stopwords=True, verbose=True):
        self.lan = language
        self.stoprm = remove_stopwords
        # (Down)load stopwords for nltk
        if self.stoprm:
            nltk.download('stopwords')
            try:
                self.sw = set(stopwords.words(language.lower()))
            except OSError:
                puts_err("No stopwords for " + language + " found. Defaulting to english.")
                self.sw = set(stopwords.words("english"))

    def preprocess(self, sentence, join_list=False):
        tmp = []
        for w in sentence.split():
            if self.stoprm:
                if w not in self.sw:
                    tmp.append(w.translate(str.maketrans('', '', string.punctuation)).lower())
            else:
                tmp.append(w.translate(str.maketrans('', '', string.punctuation)).lower())
        if join_list:
            tmp = " ".join(tmp)
        return tmp


class PlaceboPreprocessor:
    """"Can be passed to Embedder if no preprocessing is required"""

    def __init__(self, language, remove_stopwords=True, verbose=True):
        self.lan = language
        self.stoprm = remove_stopwords

    def preprocess(self, sentence, join_list=False):
        if join_list:
            sentence = " ".join(sentence)
        return sentence


class CRSumPreprocessor:

    def __init__(self, language, M=5, N=5, return_sentence=True, remove_stopwords=True):
        self.M = M
        self.N = N
        self.lan = language
        self.resent = return_sentence
        self.stoprm = remove_stopwords
        self.labels = ["stm" + str(m + 1) for m in reversed(range(M))] + ["st"] + ["stn" + str(n + 1) for n in range(N)]
        # (Down)load stopwords for nltk
        if self.stoprm:
            nltk.download('stopwords')
            try:
                self.sw = set(stopwords.words(language.lower()))
            except OSError:
                puts_err("No stopwords for " + language + " found. Defaulting to english.")
                self.sw = set(stopwords.words("english"))

    def preprocess(self, sentence, join_list=False):

        def process_sentence(snt, rmstop, stopw, tfd, dfd, jlist=False):
            tmp = []
            avg_tf, avg_df = 0, 0
            for w in snt.split():
                if rmstop:
                    if w not in stopw:
                        avg_tf += tfd[w]
                        avg_df += dfd[w]
                        tmp.append(w.translate(str.maketrans('', '', string.punctuation)).lower())
                else:
                    avg_tf += tfd[w]
                    avg_df += dfd[w]
                    tmp.append(w.translate(str.maketrans('', '', string.punctuation)).lower())
            avg_tf /= len(tmp)
            avg_df /= len(tmp)
            if jlist:
                tmp = " ".join(tmp)
            return ["<L>"] + tmp + ["<R>"], avg_tf, avg_df, len(tmp)

        tf_dict = Counter(sentence.translate(str.maketrans('', '', string.punctuation)).lower().split())
        tok_text = sent_tokenize(sentence)

        df_dict = defaultdict(lambda: 0)
        for s in tok_text:
            unique_words = set(sentence.translate(str.maketrans('', '', string.punctuation)).lower().split())
            for w in unique_words:
                df_dict[w] += 1

        prepro_data = [process_sentence(s, self.stoprm, self.sw, tf_dict, df_dict, join_list) for s in tok_text]
        prepro_text = [x[0] for x in prepro_data]
        prepro_text = [[] for x in range(self.M)] + prepro_text + [[] for x in range(self.N)]
        sent_termfreq = [x[1] for x in prepro_data]
        sent_docfreq = [x[2] for x in prepro_data]
        sent_len = [x[3] for x in prepro_data]
        sentence_df = pd.DataFrame("", index=range(len(tok_text)), columns=self.labels)

        for i, row in sentence_df.iterrows():
            for j in range(len(self.labels)):
                row[self.labels[j]] = prepro_text[j + i]

        sentence_df["len"] = sent_len
        sentence_df["pos"] = sentence_df.index / (len(sentence_df.index) - 1)
        sentence_df["df"] = sent_docfreq
        sentence_df["tf"] = sent_termfreq
        sentence_df["Language"] = self.lan
        if self.resent:
            return sentence_df, tok_text
        else:
            return sentence_df
