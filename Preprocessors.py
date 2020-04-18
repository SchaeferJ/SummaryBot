#!/usr/local/bin/python3

import nltk
import string
from nltk.corpus import stopwords
from clint.textui import puts, puts_err, prompt, colored


class StandardPreprocessor:
    """"Standard Preprocessor for sentence embedding"""

    def __init__(self, language, remove_stopwords=True, verbose=True):
        self.lan = language
        self.stoprm = remove_stopwords
        # (Down)load stopwords for nltk
        if self.stoprm:
            nltk.download('stopwords')
            try:
                self.sw = set(stopwords.words(language))
            except OSError:
                puts_err("No stopwords for " + language + " found. Defaulting to english.")
                self.sw = set(stopwords.words("english"))

    def preprocess(self, sentence):
        tmp = []
        for w in sentence.split():
            if self.stoprm:
                if w not in self.sw:
                    tmp.append(w.translate(str.maketrans('', '', string.punctuation)).lower())
            else:
                tmp.append(w.translate(str.maketrans('', '', string.punctuation)).lower())
        return " ".join(tmp)
