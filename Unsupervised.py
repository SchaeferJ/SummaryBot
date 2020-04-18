#!/usr/local/bin/python3

from nltk.tokenize import sent_tokenize
from collections import Counter
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import numpy as np
from Embedder import Embedder


class kmeansSummarizer:

    def __init__(self, embedding_model, preprocessor, seed=None):
        self.emb = Embedder(embedding_model, preprocessor)
        self.ppr = preprocessor
        self.language = embedding_model.get_language()
        self.dimensionality = embedding_model.get_dimensionality()
        # Seed for k-Means Clustering. Can be adjusted manually
        self.seed = seed

    def summarize(self, text, sum_len, sif=True, npc=3, svd_iterations=7):
        tok_text = sent_tokenize(text, self.language.lower())
        if sum_len < 1:
            sum_len = round(sum_len * len(tok_text))
        else:
            sum_len = int(sum_len)
        doc_matrix = np.zeros((len(tok_text), self.dimensionality))
        prp_text = self.ppr.preprocess(sentence=text)
        if sif:
            word_freqs = Counter(prp_text)
            for i, s in enumerate(tok_text):
                doc_matrix[i] = self.emb.embed_sentence(s, sif=True, word_freqs=word_freqs)
            svd = TruncatedSVD(n_components=npc, n_iter=svd_iterations, random_state=self.seed)
            svd.fit(doc_matrix)
            u = svd.components_
            if npc == 1:
                doc_matrix = doc_matrix - doc_matrix.dot(u.transpose()) * u
            else:
                doc_matrix = doc_matrix - doc_matrix.dot(u.transpose()).dot(u)
        else:
            for i, s in enumerate(tok_text):
                doc_matrix[i] = self.emb.embed_sentence(s, sif=False)

        print(doc_matrix.shape)
        kmeans = KMeans(n_clusters=sum_len, random_state=self.seed)
        kmeans = kmeans.fit(doc_matrix)
        avg = []
        for j in range(sum_len):
            idx = np.where(kmeans.labels_ == j)[0]
            avg.append(np.mean(idx))
        closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, doc_matrix)
        ordering = sorted(range(sum_len), key=lambda k: avg[k])
        return ' '.join([tok_text[closest[ordering[idx]]] for idx in ordering])


# Minimal working example
from Fasttext import FTEmbedder
from Preprocessors import StandardPreprocessor
from clint.textui import puts, prompt, colored

if __name__ == "__main__":
    textfile = input("Enter the path to the text file you want to summarize: ")
    with open(textfile, 'r') as file:
        longtext = file.read().replace('\n', ' ')
    l = input("Which language is the text? ")
    sentcount = float(input("How long should the summary be?\nNumber of sentences or fraction of total: "))
    fte = FTEmbedder(l)
    spp = StandardPreprocessor(l)
    summarizer = kmeansSummarizer(embedding_model=fte, preprocessor=spp)
    use_sif = prompt.yn("Do you want to use smooth inverse frequencies?")
    summary = summarizer.summarize(longtext, sentcount, sif=use_sif)
    puts("Summary:")
    puts(colored.blue(summary))
    puts("Original text:")
    puts(colored.magenta(longtext))

