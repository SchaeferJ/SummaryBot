#!/usr/local/bin/python3
"""
fastText embedder (and preprocesser) for Embedding-Instances
"""
import yaml
import requests
import os
import sys
import dill
import tqdm
import codecs
import pickle
import numpy as np

from clint.textui import puts, puts_err, prompt, colored
from collections import defaultdict


class FTEmbedder:
    """ Provides a standarized interface for fastText word embeddings. Instances of his class are meant
    to be passed to instances of the more general Embedder-Class."""

    def __init__(self, language, verbose=True, configfile="config.yml"):
        """
        :param language: str, Language, in which the words will be embedded
        :param verbose: boolean, Prints more expressive status messages when True
        :param configfile: str, path to YAML-File with configuration
        """
        with open(configfile, "r") as infile:
            cfg = yaml.load(infile, Loader=yaml.SafeLoader)

        self._fastlinks = cfg["FasttextLink"]
        self._langs = cfg["AvailabeLangs"]
        self._gencnf = cfg["General"]
        self._ft_dir = self._gencnf["FasttextDirectory"]
        self._pe_dir = self._gencnf["MatrixDirectory"]
        self._verbose = verbose

        if not os.path.isdir(self._pe_dir):
            puts("The matrix directory you've configured does not yet exist.")
            if prompt.yn("Do you want to create the directory " + colored.green(self._pe_dir) + " now?"):
                os.mkdir(self._pe_dir)
            else:
                puts_err("Please define an existing MatrixDirectory in config.yaml. Bye!")
                sys.exit()

        self.__set_language(language)

    def embed(self, word: str) -> list:
        """
        Returns the fastText-embedding vector of the given word
        :param word: str, a word
        :return: list of floats, the word embedding
        """
        return self.emb_matrix[self.wordmapper[word]]

    def get_dimensionality(self) -> int:
        """
        Returns the dimensionality of loaded embeddings
        :return: int, the dimensionality
        """
        return self.dimensionality

    def __download_pretrained(self, fname: str, fdir: str):
        """
        Downloads pre-trained fastText vectors. Download-URL and target directory are defined in the configuration file.
        :param fname: str, Name of the vector file on the server (from config.yml)
        :param fdir: str, local download path
        :return: None
        """
        download_url = self._fastlinks["url"] + fname
        r = requests.get(download_url, stream=True)
        with open(fdir, "wb") as downfile:
            total_length = int(r.headers.get('content-length'))
            tt = float("{:.2f}".format(total_length / 1024 ** 2))
            for ch in tqdm.tqdm(iterable=r.iter_content(chunk_size=1024 ** 2), total=tt, unit='MB'):
                if ch:
                    downfile.write(ch)

    def __set_language(self, lan: str):
        """
        Loads the files containing the word embeddings of the given language into memory.
        If the necessary files are missing, they are created from the fastText vector files.
        After the files have been loaded, words can be embedded in the chosen language
        :param lan: str, the language for which embeddings are loaded
        :return: None
        """
        # Check that pre-trained fastText embeddigns exist
        if lan not in self._langs:
            raise Exception(lan + " is not a supported Language")

        self.wm_fname = "wordmapper_" + lan + ".dill"
        self.wm_fdir = os.path.join(self._pe_dir, self.wm_fname)

        self.mat_fname = "emb_matrix_" + lan + ".pck"
        self.mat_fdir = os.path.join(self._pe_dir, self.mat_fname)

        # Create the required files (matrix + mapping dict) if not present
        if not os.path.isfile(self.mat_fdir) or not os.path.isfile(self.wm_fdir):
            self.__init_language(lan)

        if self._verbose:
            puts("Loading embeddings for " + lan)

        with open(self.wm_fdir, 'rb') as file:
            self.wordmapper = dill.load(file)

        with open(self.mat_fdir, 'rb') as file:
            self.emb_matrix = pickle.load(file)

        self.dimensionality = len(self.emb_matrix[0])

        if self._verbose:
            puts("Done.")

    def __init_language(self, lan: str):
        """
        Converts a fastText text vector file (xy.vec) to the format required for embedding: A numpy matrix with one
        row per word and a dictionary mapping the words to the index of their embedding in the matrix.
        Matrix and dictionary are stored on disk.
        if the required fastText vectors are missing, the user is prompted whether they should be downloaded.

        :param lan: str, the language in which the words will be embedded
        :return: None
        """

        # Construct download URL and download directory
        vec_fname = ".".join([self._fastlinks["prefix"], self._langs[lan], self._fastlinks["suffix"]])
        vec_fdir = os.path.join(self._ft_dir, vec_fname)

        if not os.path.isdir(self._ft_dir):
            puts("The fastText directory you've configured does not yet exist.")
            if prompt.yn("Do you want to create the directory " + colored.green(self._ft_dir) + " now?"):
                os.mkdir(self._ft_dir)
            else:
                puts_err("Please define an existing FasttextDirectory in config.yaml. Bye!")
                sys.exit()

        # Download required fastText vectors if not present
        if not os.path.isfile(vec_fdir):
            puts("No fastText-Vectors found for language " + lan + ".")
            puts("You can download pre-trained vectors from Facebook.\nThis will take up to 12 GB of space.")
            if prompt.yn("Do you want to download the pre-trained vectors for " + colored.green(lan) + " now?"):
                self.__download_pretrained(vec_fname, vec_fdir)
                puts("Download complete.")
            else:
                puts_err("Please place appropriate vectors in your vector directory and restart. Bye!")
                sys.exit()
        if self._verbose:
            puts("Loading vectors. This may take a while...")

        # Read and split the embeddings => One word-embedding-pair per line in .vec file.
        lines = codecs.open(vec_fdir, "r", "UTF-8").read().splitlines()
        if self._verbose:
            puts("Done.")
        # Determine vocabulary size (number of embeddings in file) and dimensionality (length of individual embedding
        # vectors)
        vocab_size = int(lines[0].split()[0])
        self.dimensionality = int(lines[0].split()[1])
        if self._verbose:
            puts("Found " + str(self.dimensionality) + "-dimensional vocabulary of size " + str(vocab_size) + ".")
            puts("Parsing vector file. Please stand by.")
        # Separate word from its embedding
        word_list = []
        for l in tqdm.tqdm(lines[1:], unit="Words"):
            # Individual values are separated by whitespace
            linelist = l.split(" ")
            # First entry in line is the word
            word = linelist[0]
            # Remaining entries are the embedding => Must be converted to float
            vec = linelist[1:]
            vec = [float(i) for i in vec]
            # Some of the fastText files are known to be corrupted, skip faulty lines
            if len(vec) != self.dimensionality:
                puts_err("Error: Invalid vector for " + word + " Skipping")
            else:
                word_list.append((word, vec))
        if self._verbose:
            puts("Done.")
            puts("Calculating average word vector for out-of-vocabulary words.")
        # Calculate embedding of out-of-vocabulary words as mean of all other embeddings.
        oov_vec = np.array([vec for _, vec in word_list]).mean(axis=0)
        if self._verbose:
            puts("Done")
        # Create dict mapping word to row of embedding matrix (default, i.e. OOV-word is 0)
        wordmapper = defaultdict(lambda: 0)
        emb_matrix = np.zeros((len(word_list) + 1, self.dimensionality))
        if self._verbose:
            puts("Converting vectors to matrix. Please stand by.")
        i = 1
        # Load words and embeddings to dict/matrix
        for word_tup in tqdm.tqdm(word_list):
            word, embedding = word_tup
            wordmapper[word] = i
            emb_matrix[i] = embedding
            i += 1
        emb_matrix[0] = oov_vec
        if self._verbose:
            puts("Done.")
            puts("Saving matrix.")
        # Save matrix and dict
        # NOTE: defaultdicts cannot be pickled, therefore they must be dill-ed.
        # NOTE ALSO: embedding matrix can be very large. Pickle protocol 4 must be used.
        with open(self.wm_fdir, 'wb') as file:
            dill.dump(wordmapper, file)
        with open(self.mat_fdir, 'wb') as file:
            pickle.dump(emb_matrix, file, protocol=4)
        if self._verbose:
            puts("Language " + lan + " initialized successfully.")
        if prompt.yn("The file " + colored.green(vec_fname) + " is no longer needed. Do you want to delete it now?"):
            os.remove(vec_fdir)


# Minimal working example
if __name__ == "__main__":
    l = input("Enter a language: ")
    emb = FTEmbedder(l)
    w = input("Enter a word: ")
    print("The embedding of " + w + " is:")
    print(emb.embed(w))
