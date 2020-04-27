#!/usr/local/bin/python3

import mysql.connector as mysql
import pandas as pd
from nltk.tokenize import sent_tokenize
from Preprocessors import *
from Fasttext import FTEmbedder
from Embedder import Embedder
import tqdm
import random
from collections import Counter, defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import pickle
from clint.textui import puts


host = input("Please enter database hostname:\n")
dbname = input("Please enter database name:\n")
uname = input("Please enter database username:\n")
pw = input("Please enter database password:\n")

puts("Connecting to Database")
db = mysql.connect(
    host=host,
    user=uname,
    passwd=pw,
    database=dbname
)
puts("Done.")


cursor = db.cursor()
query = "SELECT Publisher, Language, Lead, Body FROM Article JOIN Publisher ON Article.Publisher = Publisher.Name WHERE Lead <>''"
puts("Retrieving data")
cursor.execute(query)
result = cursor.fetchall()
# Fetch Data
article_list = []
for r in result:
    article_list.append(r)
db.close()
puts("Done.")
article_df = pd.DataFrame(article_list, columns=["Publisher", "Language", "Lead", "Body"])

article_df = article_df.reset_index()
article_df['ID'] = article_df.index

traincount = int(len(article_df.index) * 0.8)
testcount = len(article_df.index) - traincount
testsplit = [True] * traincount + [False] * testcount
for i in range(random.SystemRandom().randint(1, 10)):
    random.shuffle(testsplit)

article_df["isTrain"] = testsplit

puts("Initializing language-specific components")
lang_pprs = {}
lang_embedders = {}
lang_sw = {}
for l in article_df.Language.unique():
    lang_pprs[l] = CRSumPreprocessor(l, 5, 5, return_sentence=True)
    lang_embedders[l] = Embedder(embedding_model=FTEmbedder(l), preprocesser=PlaceboPreprocessor(l))
    lang_sw[l] = set(stopwords.words(l.lower()))
puts("Done.")

prepro_frames = []

for ind, row in tqdm.tqdm(article_df.iterrows(), total=len(article_df.index), unit="Articles"):
    cosine_sims = []
    tmp, sents = lang_pprs[row.Language].preprocess(row.Body)
    sum_nsw = [w for w in row.Lead.split() if not w in lang_sw[row.Language]]
    sum_embedding = lang_embedders[row.Language].embed_sentence(sum_nsw, sif=False)
    for s in sents:
        sent_nsw = [w for w in s.split() if not w in lang_sw[row.Language]]
        sent_embedding = lang_embedders[row.Language].embed_sentence(sent_nsw, sif=False)
        cosine_sims.append(cosine_similarity(sum_embedding.reshape(1, 300), sent_embedding.reshape(1, 300))[0][0])
    tmp["sen"] = tmp.index
    tmp["ID"] = ind
    tmp["isTrain"] = row.isTrain
    tmp["cosine_sim"] = cosine_sims
    prepro_frames.append(tmp)

processed_articles = pd.concat(prepro_frames)
processed_articles = processed_articles.reset_index()
processed_articles = processed_articles[["ID", "sen", "stm5", "stm4", "stm3", "stm2", "stm1", "st", "stn1", "stn2", "stn3", "stn4", "stn5", "len", "pos", "tf", "df", "cosine_sim", "Language", "isTrain"]]
train_df = processed_articles[processed_articles.isTrain == True]
test_df = processed_articles[processed_articles.isTrain == False]
del train_df["isTrain"]
del test_df["isTrain"]
scaler = MinMaxScaler()
train_df[['len', 'df','tf']] = scaler.fit_transform(train_df[['len', 'df','tf']])

with open("./training_data/scaler.pkl", 'wb') as file:
    pickle.dump(scaler, file)

train_df.to_pickle('./training_data/train_set.pkl')
test_df.to_pickle('./training_data/test_set.pkl')