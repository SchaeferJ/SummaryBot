import nltk
import string
from nltk.corpus import stopwords
from clint.textui import puts, puts_err, prompt, colored


class StandardPreprocessor:
    """"Standard Preprocessor for sentence embedding"""

    def __init__(self, language, verbose=True):
        self.lan = language
        nltk.download('stopwords')
        try:
            self.sw = set(stopwords.words(language))
        except OSError:
            puts_err("No stopwords for " + language + " found. Defaulting to english.")
            self.sw = set(stopwords.words("english"))

    def preprocess(self, sentence):
        tmp = []
        for w in sentence.split():
            if not w in self.stopWords:
                tmp.append(w.translate(str.maketrans('', '', string.punctuation)).lower())
        return " ".join(tmp)
