#!/usr/local/bin/python3

from nltk.tokenize import sent_tokenize
from collections import Counter
from sklearn.decomposition import TruncatedSVD

class kmeansSummarizer:

    def __init__(self, embedder, preprocessor, seed=None):
        self.emb = embedder
        self.ppr = preprocessor
        self.language = embedder.get_language()
        # Seed for k-Means Clustering. Can be adjusted manually
        self.seed = seed

    def summarize(self, text, sum_len, sif=True, npc=3, svd_iterations=7, a=10**-3):
        if sum_len < 1:
            sum_len = round(sum_len * len(text))
        doc_matrix = np.zeros((len(text_nopunct), dimensionality))
        tok_text = sent_tokenize(text, self.language.lower())
        prp_text = self.ppr.preprocess(tok_text)
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
                doc_matrix[i] = self.emb.embed_sentence(s, sif=True, word_freqs=word_freqs)

        kmeans = KMeans(n_clusters=sum_len, random_state=4711)
        kmeans = kmeans.fit(doc_matrix)
        avg = []
        closest = []
        for j in range(sum_len):
            idx = np.where(kmeans.labels_ == j)[0]
            avg.append(np.mean(idx))
        closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, doc_matrix)
        ordering = sorted(range(sum_len), key=lambda k: avg[k])
        return ' '.join([text[closest[ordering[idx]]] for idx in ordering])

