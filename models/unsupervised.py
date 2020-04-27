#!/usr/local/bin/python3
"""
Unsupervised Methods for extractive summarization
"""
from nltk.tokenize import sent_tokenize
from collections import Counter
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import numpy as np
from Embedder import Embedder


class kMeans:
    """ Generates extractive summaries of texts by performing k-means clustering. For a summary of length x,
    x clusters are computed and the centroids of each cluster are added to the summary """
    def __init__(self, embedding_model, preprocessor, seed=None):
        """
        Creates new k-Means summarizer object
        :param embedding_model: The embedding model to be used (e.g. FTEmbedder)
        :param preprocessor: The preprocessor to be used
        :param seed: optional, seed for random number generators
        """

        self.embedding_model = embedding_model
        self.preprocessor = preprocessor

        self.emb = None
        self.ppr = None
        self.language = None
        self.embedding_instance = None
        self.dimensionality = None
        # Seed for k-Means Clustering. Can be adjusted manually
        self.seed = seed

    def summarize(self, text: str, language:str, sum_len, sif=True, npc=3, svd_iterations=7) -> str:
        """
        Computes and returns an extractive summary of a text using k-Means clustering
        :param text: str, the text to be summarized
        :param language: str, the language of the text
        :param sum_len: int/float, length of summary. Int>1: Number of sentences, float<1: fraction of original length
        :param sif: boolean, applies smooth inverse frequency aggregation when True
        :param npc: int, optional: number of principal components for truncated SVD of SIF
        :param svd_iterations: int, optional: number of iterations for truncated SVD of SIF
        :return: str, the summary
        """

        if self.language is None or self.language != language:
            self.embedding_instance = self.embedding_model(language)
            self.ppr = self.preprocessor(language)
            self.dimensionality = self.embedding_instance.get_dimensionality()
            self.emb = Embedder(self.embedding_instance, self.ppr)
            self.language = language





        # Split text into sentences
        tok_text = sent_tokenize(text, self.language.lower())
        # If length is smaller than 1, compute the number of sentences the fraction corresponds to
        if sum_len < 1:
            sum_len = round(sum_len * len(tok_text))
        else:
            sum_len = int(sum_len)
        doc_matrix = np.zeros((len(tok_text), self.dimensionality))
        if sif:
            # Split text into individual words and count their frequencies for SIF
            prp_text = self.ppr.preprocess(sentence=text)
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

        #print(doc_matrix.shape)
        kmeans = KMeans(n_clusters=sum_len, random_state=self.seed)
        kmeans = kmeans.fit(doc_matrix)
        avg = []
        for j in range(sum_len):
            idx = np.where(kmeans.labels_ == j)[0]
            avg.append(np.mean(idx))
        closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, doc_matrix)
        closest.sort()
        return ' '.join([tok_text[idx] for idx in closest])

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
    summarizer = kMeans(FTEmbedder, StandardPreprocessor)
    use_sif = prompt.yn("Do you want to use smooth inverse frequencies?")
    summary = summarizer.summarize(longtext, l, sentcount, sif=use_sif)
    puts("Summary:")
    puts(colored.blue(summary))
    puts("Original text:")
    puts(colored.magenta(longtext))

