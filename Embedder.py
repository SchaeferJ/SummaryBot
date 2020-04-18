#!/usr/local/bin/python3

import numpy as np


class Embedder:

    def __init__(self, embedder, preprocesser, sif=True):
        self.emb = embedder
        self.ppr = preprocesser
        self._smoothing_factor = 10 ** -3

    def set_smoothing_factor(self, sf):
        self._smoothing_factor = sf

    def embed_sentence(self, sentence, word_freqs=None):
        if self.sif:
            return self._sif_sent_emb(sentence, word_freqs)
        else:
            return self._mean_sent_emb(sentence)

    def _mean_sent_emb(self, sentence):
        words = sentence.split()
        sent_emb = np.zeros(self.emb.dimensionality)
        for w in words:
            sent_emb += self.emb.embed(w)
        sent_emb /= len(words)
        return sent_emb

    def _sif_sent_emb(self, sentence, word_freqs):
        words = sentence.split()
        sent_emb = np.zeros(self.emb.dimensionality)
        for w in words:
            sent_emb += self._smoothing_factor / (self._smoothing_factor + word_freqs[w]) * self.emb.embed(w)
        sent_emb /= len(words)
        return sent_emb