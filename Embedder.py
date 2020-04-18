#!/usr/local/bin/python3

import yaml
import requests
import os
import sys

from clint.textui import progress, prompt, colored

class Embedder:
    def __init__(self, configfile="config.yml"):
        with open(configfile, "r") as infile:
            cfg = yaml.load(infile, Loader=yaml.SafeLoader)

        self.fastlinks = cfg["FasttextLink"]
        self.langs = cfg["AvailabeLangs"]
        self.gencnf = cfg["General"]

    def download_pretrained(self, fname):
        download_url = self.fastlinks["url"]+fname
        print(download_url)

    def setLanguage(self, lan: str):
        if lan not in self.langs:
            raise Exception(lan+" is not a supported Language")
        vec_filename = ".".join([self.fastlinks["prefix"], self.langs[lan], self.fastlinks["suffix"]])
        if not os.path.isfile(os.path.join(self.gencnf["FasttextDirectory"], vec_filename)):
            print("No fastText-Vectors found for language "+lan+".")
            print("You can download pre-trained vectors from Facebook. This will take up to 6 GB of space.")
            if prompt.yn("Do you want to download the pre-trained vectors for " + colored.blue(lan) + " now?"):
                self.download_pretrained(vec_filename)
            else:
                print("Please place appropriate vectors in your vector directory and restart. Bye!")
                sys.exit()
        print(vec_filename)


if __name__ == "__main__":
    emb = Embedder()
    print(emb.fastlinks)
    emb.setLanguage("English")
