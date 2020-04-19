#!/usr/local/bin/python3
# coding: utf-8

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

M, N = 5, 5


def get_sen_labels():
    sen_labels = []
    for i in reversed(range(M)):
        sen_labels.append('stm' + str(i + 1))
    sen_labels.append('st')
    for i in range(N):
        sen_labels.append('stn' + str(i + 1))
    return sen_labels

def padding(sentence):
    sentence.insert(0, '<L>')
    sentence.append('<R>')
    return sentence

def get_tf(row):
    tf, word2tf = 0, id_word2tf[row.id]
    for word in row.st:
        tf += word2tf[word]
    return tf / len(row.st)

def get_df(row):
    rowlang = article_df[article_df.ID == int(row.id)].Language.to_string(index=False).strip()
    df = 0
    for word in row.st:
        df += lang_dfdicts[rowlang][word]
    return df / len(row.st)

def bigramize(sentence):
    return set([b for b in zip(sentence[:-1], sentence[1:])])

def rouge_2(row):
    if len(row.st) < 2:
        return 0
    else:
        st_bigrams = bigramize(row.st)
        summary_bigrams = summary_bigrams_df.loc[summary_bigrams_df.ID == row.id].summary_words.values[0]
        overlap = len(st_bigrams.intersection(summary_bigrams))
        if (len(summary_bigrams)) > 0:
            return overlap / len(summary_bigrams)
        else:
            return 0


def make_sent_emb(words):
    sent_emb = np.zeros(300)
    for w in words:
        if w not in ["<L>", "<R>"]:
            sent_emb += emb_matrix[wordmapper[w]]
    sent_emb /= len(words)
    return sent_emb

def get_cosim(row):
    rowlang = article_df[article_df.ID == int(row.id)].Language.to_string(index=False).strip()
    wf_summary = Counter(article_df.loc[article_df.ID == row.id].summary_words.values[0])
    summary = lang_embedders[rowlang].embed_sentence(article_df.loc[article_df.ID == row.id].summary_words.values[0],
                                                     sif=True,
                                                     word_freqs=wf_summary)
    wf_sent = Counter(row.st[1:-1])
    sent = lang_embedders[rowlang].embed_sentence(row.st[1:-1], sif=True,
                                                  word_freqs=wf_sent)
    # summary = make_sent_emb(article_df.loc[article_df.ID == row.id].summary_words.values[0])
    # sent = make_sent_emb(row.st)
    return cosine_similarity(summary.reshape(1, 300), sent.reshape(1, 300))[0][0]


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
lang_dfdicts = {}
lang_casecount = {}
lang_embedders = {}
for l in article_df.Language.unique():
    lang_pprs[l] = StandardPreprocessor(l.lower())
    lang_dfdicts[l] = defaultdict(lambda: 0)
    lang_casecount[l] = sum(article_df[article_df.isTrain].Language == l)
    lang_embedders[l] = Embedder(embedding_model=FTEmbedder(l), preprocesser=PlaceboPreprocessor(l))
puts("Done.")

doc_sentences = []
doc_words = []
summary_words = []
id_word2tf = {}
puts("Preprocessing articles. Please stand by.")
for _, row in tqdm.tqdm(article_df.iterrows(), total=len(article_df.index)):
    rowlang = row["Language"]
    sentences = sent_tokenize(row["Body"], rowlang.lower())
    sentence_words = [lang_pprs[rowlang].preprocess(s) for s in sentences]
    sentences_pp = [" ".join(s) for s in sentence_words]
    doc_sentences.append(sentences_pp)
    doc_words.append(sentence_words)
    # summ = sent_tokenize(row["Lead"],rowlang.lower())
    summ_words = lang_pprs[rowlang].preprocess(row["Lead"])
    summary_words.append(summ_words)

    total_words = [word for sent in sentence_words for word in sent]
    text_length = len(total_words)
    wordcount = Counter(total_words)
    for word in wordcount:
        wordcount[word] /= text_length
        if row.isTrain:
            lang_dfdicts[rowlang][word] += 1
    id_word2tf[row.ID] = wordcount

puts("Done.")

for lng in lang_dfdicts:
    for word in lang_dfdicts[lng]:
        lang_dfdicts[lng][word] /= lang_casecount[l]


del lang_pprs

article_df["document_sentences"] = doc_sentences
article_df["document_words"] = doc_words
article_df["summary_words"] = summary_words

puts("Transforming articles to required format. Please stand by.")
id_list, sen_list, sentences_list, pos_list = [], [], [], []
for _, row in tqdm.tqdm(article_df.iterrows(), total=len(article_df.index)):
    id, document = row.ID, row.document_words
    doc_len = len([word for sen in document for word in sen])

    word_cursor = 0
    for sentence in document:
        word_cursor += len(sentence)
        pos_list.append(word_cursor / doc_len)

    document = list(map(padding, document))
    for _ in range(M):
        document.insert(0, [])
    for _ in range(N):
        document.append([])

    for i in range(len(document) - M - N):
        id_list.append(row.ID)
        sen_list.append(i)
        sentences_list.append(document[i:i + M + N + 1])

puts("Done")

id_df = pd.DataFrame(id_list)
sen_df = pd.DataFrame(sen_list)
sentences_df = pd.DataFrame(sentences_list)
pos_df = pd.DataFrame(pos_list)

sentence_df = pd.concat([id_df, sen_df, sentences_df], axis=1)
sentence_df.columns = ['id', 'sen'] + get_sen_labels()

summary_bigrams_df = article_df[['ID', 'summary_words']]
summary_bigrams_df["summary_words"] = summary_bigrams_df.summary_words.apply(bigramize)

len_df = sentence_df.st.apply(lambda x: len(x) - 2)
puts("Calculating TF scores")
tf_df = sentence_df.apply(get_tf, axis=1)
puts("Calculating DF scores")
df_df = sentence_df.apply(get_df, axis=1)
puts("Calculating cosine similarities")
cos_df = sentence_df.apply(get_cosim, axis=1)
puts("Calculating ROUGE scores")
rouge_2_df = sentence_df.apply(rouge_2, axis=1)
puts("Done")

prepro_df = pd.concat([sentence_df, len_df, pos_df, tf_df, df_df, rouge_2_df, cos_df], axis=1)
prepro_df.columns = ['id', 'sen'] + get_sen_labels() + ['len', 'pos', 'tf', 'df', 'rouge_2', "cosine_sim"]

lang_df = article_df[["ID", "Language"]]
prepro_df = pd.merge(prepro_df, lang_df, left_on="id", right_on="ID")

train_ids = article_df[article_df.isTrain].ID.to_list()

test_ids = article_df[article_df.isTrain == False].ID.to_list()

test_df = prepro_df[prepro_df.id.isin(test_ids)]

train_df = prepro_df[prepro_df.id.isin(train_ids)]

len_scaler = MinMaxScaler()
train_df.len = len_scaler.fit_transform(train_df.len.values.reshape(-1, 1))

test_df.len = len_scaler.transform(test_df.len.values.reshape(-1, 1))


with open("./training_data/len-scaler.pkl", 'wb') as file:
    pickle.dump(len_scaler, file)

train_df.to_pickle('./training_data/train_set.pkl')
test_df.to_pickle('./training_data/test_set.pkl')
article_df[['ID', 'summary_words']].to_pickle("./training_data/summaries.pkl")
puts("Finished!")