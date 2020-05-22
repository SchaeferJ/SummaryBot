from sklearn.metrics.pairwise import manhattan_distances, cosine_similarity, euclidean_distances

from encoders.use import USEEmbedder


class USEevaluator:

    def __init__(self, metric="cosine"):
        self.embedder = USEEmbedder("NA")

        if metric == "cosine":
            self.distmet = cosine_similarity
        elif metric == "euclidean":
            self.distmet = euclidean_distances
        elif metric == "l1":
            self.distmet = manhattan_distances

    def compare(self, text1:str, text2:str) -> float:
        emb1 = self.embedder.embed(text1)
        emb2 = self.embedder.embed(text2)
        return self.distmet(emb1, emb2)[0][0]


# Minimal working example
if __name__ == "__main__":
    comparator = USEevaluator()
    a = input("Enter a sentence: ")
    b = input("Enter another sentence: ")
    sim = comparator.compare(a,b)
    print("Cosine similarity of the two USE-Encoded sentences is "+ str(sim))