#!/usr/local/bin/python3

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
    def __init__(self, configfile="config.yml"):
        with open(configfile, "r") as infile:
            cfg = yaml.load(infile, Loader=yaml.SafeLoader)

        self.fastlinks = cfg["FasttextLink"]
        self.langs = cfg["AvailabeLangs"]
        self.gencnf = cfg["General"]
        self.ft_dir = self.gencnf["FasttextDirectory"]
        self.pe_dir = self.gencnf["MatrixDirectory"]

        if not os.path.isdir(self.pe_dir):
            puts("The matrix directory you've configured does not yet exist.")
            if prompt.yn("Do you want to create the directory " + colored.green(self.pe_dir) + " now?"):
                os.mkdir(self.pe_dir)
            else:
                puts_err("Please define an existing MatrixDirectory in config.yaml. Bye!")
                sys.exit()

    def download_pretrained(self, fname, fdir):
        download_url = self.fastlinks["url"] + fname
        r = requests.get(download_url, stream=True)
        with open(fdir, "wb") as downfile:
            total_length = int(r.headers.get('content-length'))
            tt = float("{:.2f}".format(total_length / 1024 ** 2))
            for ch in tqdm.tqdm(iterable=r.iter_content(chunk_size=1024 ** 2), total=tt, unit='MB'):
                if ch:
                    downfile.write(ch)

    def setLanguage(self, lan: str):
        if lan not in self.langs:
            raise Exception(lan + " is not a supported Language")

        self.wm_fname = "wordmapper_" + lan + ".dill"
        self.wm_fdir = os.path.join(self.pe_dir, self.wm_fname)

        self.mat_fname = "emb_matrix_" + lan + ".pck"
        self.mat_fdir = os.path.join(self.pe_dir, self.mat_fname)

        if not os.path.isfile(self.mat_fdir) or not os.path.isfile(self.wm_fdir):
            self._initLanguage(lan)

    def _initLanguage(self, lan):
        vec_fname = ".".join([self.fastlinks["prefix"], self.langs[lan], self.fastlinks["suffix"]])
        vec_fdir = os.path.join(self.ft_dir, vec_fname)

        if not os.path.isdir(self.ft_dir):
            puts("The fastText directory you've configured does not yet exist.")
            if prompt.yn("Do you want to create the directory " + colored.green(self.ft_dir) + " now?"):
                os.mkdir(self.ft_dir)
            else:
                puts_err("Please define an existing FasttextDirectory in config.yaml. Bye!")
                sys.exit()

        if not os.path.isfile(vec_fdir):
            puts("No fastText-Vectors found for language " + lan + ".")
            puts("You can download pre-trained vectors from Facebook.\nThis will take up to 12 GB of space.")
            if prompt.yn("Do you want to download the pre-trained vectors for " + colored.green(lan) + " now?"):
                self.download_pretrained(vec_fname, vec_fdir)
                puts("Download complete.")
            else:
                puts_err("Please place appropriate vectors in your vector directory and restart. Bye!")
                sys.exit()
        puts("Loading vectors. This may take a while...")
        lines = codecs.open(vec_fdir, "r", "UTF-8").read().splitlines()
        puts("Done.")
        vocab_size = int(lines[0].split()[0])
        dimensionality = int(lines[0].split()[1])
        puts("Found " + str(dimensionality) + "-dimensional vocabulary of size " + str(vocab_size) + ".")
        puts("Parsing vector file. Please stand by.")
        word_list = []
        for l in tqdm.tqdm(lines[1:], unit="Words"):
            linelist = l.split(" ")
            word = linelist[0]
            vec = linelist[1:]
            vec = [float(i) for i in vec]
            if len(vec) != dimensionality:
                puts_err("Error: Invalid vector for " + word + " Skipping")
            else:
                word_list.append((word, vec))
        puts("Done.")
        puts("Calculating average word vector for out-of-vocabulary words.")
        oov_vec = np.array([vec for _, vec in word_list]).mean(axis=0)
        puts("Done")
        wordmapper = defaultdict(lambda: 0)
        emb_matrix = np.zeros((len(word_list) + 1, dimensionality))
        puts("Converting vectors to matrix. Please stand by.")
        i = 1
        for word_tup in tqdm.tqdm(word_list):
            word, embedding = word_tup
            wordmapper[word] = i
            emb_matrix[i] = embedding
            i += 1
        emb_matrix[0] = oov_vec
        puts("Done.")
        puts("Saving matrix.")
        with open(self.wm_fdir, 'wb') as file:
            dill.dump(wordmapper, file)
        with open(self.mat_fdir, 'wb') as file:
            pickle.dump(emb_matrix, file, protocol=4)
        puts("Language "+lan+" initialized successfully.")
        if prompt.yn("The file " + colored.green(vec_fname) + " is no longer needed. Do you want to delete it now?"):
            os.remove(vec_fdir)


if __name__ == "__main__":
    l = input("Language: ")
    emb = FTEmbedder()
    emb.setLanguage(l)
