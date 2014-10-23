# !/usr/bin/python
# -*- coding: utf-8 -*-

import sklearn.feature_extraction as fe
import nltk
from scipy import *
from scipy.sparse import *


def build_tdm(filenames):
    texts = [open(fn, "r+").read() for fn in filenames]
    stopwords = nltk.corpus.stopwords.words('english')
    vectorizer = fe.text.CountVectorizer(min_df=1, ngram_range=(1, 2), max_df=0.5, stop_words=stopwords)

    # строки -- документы, столбцы -- слова
    X_words = vectorizer.fit_transform(texts)
    return vectorizer.get_feature_names(), X_words

tdm = build_tdm(["artm/book.txt", "artm/cartoon.txt", "artm/series.txt"])[1]


def em(tdm, topics, iterations):
    docs, terms = tdm.shape[0]
    theta = csc_matrix((terms, topics))
    for i in xrange(iterations):
        print i, "iter"
        #     init shit
        for doc in xrange(docs):
            for term in xrange(terms):
                print theta[term]



    return 0

print em(tdm, 2, 20)