import tensorflow_hub as hub
import numpy as np
import tensorflow_text
import tensorflow as tf
import yaml

class USEEmbedder:
    """ Provides a standarized interface for fastText word embeddings. Instances of his class are meant
    to be passed to instances of the more general Embedder-Class."""

    def __init__(self, language: str, verbose=True, configfile="config.yml"):

        with open(configfile, "r") as infile:
            cfg = yaml.load(infile, Loader=yaml.SafeLoader)

        self.uselink = cfg["USELink"]
        #print(self.uselink)
        self.tfmodel = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")
        self.dimensionality = 512
        self.language_dependent = False
        self.requires_prepro = False

    def embed(self, word: str):
        return self.tfmodel(word).numpy()

    def get_dimensionality(self) -> int:
        return self.dimensionality