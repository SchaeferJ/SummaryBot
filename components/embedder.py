#!/usr/local/bin/python3
"""General Class that provides an Interface for embedding sentences"""

import numpy as np


class Embedder:
    """Sentence embedder"""

    def __init__(self, embedding_model, preprocesser):
        self.emb = embedding_model
        self.ppr = preprocesser
        self._smoothing_factor = 10 ** -3
        self.dimensionality = embedding_model.get_dimensionality()

    def set_smoothing_factor(self, sf):
        """
        Sets smoothing factor (a.k.a. 'a') of smooth inverse frequency aggregation
        :param sf: smoothing factor
        :return: None
        """
        self._smoothing_factor = sf

    def embed_sentence(self, sentence: str, sif=False,  word_freqs=None):
        """
        Returns a sentence embedding using the specified aggregation mehod
        :param sentence: str, a sentence
        :param sif: boolean, apply smooth inverse frequencies when True
        :param word_freqs: dict, {word:document frequency}, required for SIF aggregation only
        :return: ndarray, the sentence embedding
        """
        sp = self.ppr.preprocess(sentence)
        if sif:
            return self._sif_sent_emb(sp, word_freqs)
        else:
            return self._mean_sent_emb(sp)

    def _mean_sent_emb(self, sentence):
        """
        Calculates sentence embedding as simple mean of individual words
        :param sentence: str, a sentence
        :return: ndarray, the sentence embedding
        """
        sent_emb = np.zeros(self.dimensionality)
        for w in sentence:
            sent_emb += self.emb.embed(w)
        sent_emb /= len(sentence)
        return sent_emb

    def _sif_sent_emb(self, sentence, word_freqs):
        """
        Calculates sentence embedding as smooth-inverse-frequency-weighted average of words
        :param sentence: str, a sentence
        :param word_freqs: dict, {word:document frequency}, required for SIF aggregation only
        :return: ndarray, the sentence embedding
        """
        #   words = sentence.split()
        sent_emb = np.zeros(self.dimensionality)
        for w in sentence:
            sent_emb += self._smoothing_factor / (self._smoothing_factor + word_freqs[w]) * self.emb.embed(w)
        sent_emb /= len(sentence)
        return sent_emb


# Minimal working example
from encoders.fasttext import FTEmbedder
from components.preprocessors import StandardPreprocessor

if __name__ == "__main__":
    l = input("Enter a language: ")
    fte = FTEmbedder(l)
    spp = StandardPreprocessor(l)
    emb = Embedder(embedding_model=fte, preprocesser=spp)
    w = input("Enter a sentence: ")
    print("The embedding of " + w + " is:")
    print(emb.embed_sentence(w, sif=False))
