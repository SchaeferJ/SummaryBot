#!/usr/local/bin/python3

import nltk
import string
from nltk.corpus import stopwords
from clint.textui import puts_err


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