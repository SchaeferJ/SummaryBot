#!/usr/local/bin/python3

import yaml
import requests
import os
import sys
import tqdm

from clint.textui import progress, prompt, colored


class FTEmbedder:
    def __init__(self, configfile="config.yml"):
        with open(configfile, "r") as infile:
            cfg = yaml.load(infile, Loader=yaml.SafeLoader)

        self.fastlinks = cfg["FasttextLink"]
        self.langs = cfg["AvailabeLangs"]
        self.gencnf = cfg["General"]
        self.ft_dir = self.gencnf["FasttextDirectory"]

        if not os.path.isdir(self.ft_dir):
            print("The fastText directory you've configured does not yet exist.")
            if prompt.yn("Do you want to create the directory " + colored.green(self.ft_dir) + " now?"):
                os.mkdir(self.ft_dir)
            else:
                print("Please define an existing FasttextDirectory in config.yaml. Bye!")
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

        vec_filename = ".".join([self.fastlinks["prefix"], self.langs[lan], self.fastlinks["suffix"]])
        vec_fdir = os.path.join(self.ft_dir, vec_filename)

        if not os.path.isfile(vec_fdir):
            print("No fastText-Vectors found for language " + lan + ".")
            print("You can download pre-trained vectors from Facebook.\nThis will take up to 6 GB of space.")
            if prompt.yn("Do you want to download the pre-trained vectors for " + colored.green(lan) + " now?"):
                self.download_pretrained(vec_filename, vec_fdir)
            else:
                print("Please place appropriate vectors in your vector directory and restart. Bye!")
                sys.exit()
        print(vec_filename)


if __name__ == "__main__":
    l = input("Language:")
    emb = Embedder()
    emb.setLanguage(l)
