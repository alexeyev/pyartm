# !/usr/bin/python
# -*- coding: utf-8 -*-

import nltk
from scipy import *
from scipy.sparse import *
import sklearn.feature_extraction as fe


def build_tdm(filenames, min_df=0.1, max_df=0.3, ngram_range=(1, 1)):
    """
        Given filenames, builds term-document matrix;
        rows = terms, columns = documents
    """
    texts = [open(fn, "r+").read() for fn in filenames]
    stopwords = nltk.corpus.stopwords.words('english')
    vectorizer = fe.text.CountVectorizer(min_df=min_df, ngram_range=ngram_range, max_df=max_df, stop_words=stopwords)

    X_words = vectorizer.fit_transform(texts).transpose()

    for i in xrange(len(vectorizer.get_feature_names())):
        print vectorizer.get_feature_names()[i], "\t\t\t",
        print X_words[i, :].todense()
    return vectorizer.get_feature_names(), X_words


def init_matrices(terms, docs, topics):
    """
        Initiializes Phi and Theta
        phi_w,t = p(w | t), theta_t,d = p(t | d)
    """
    phi = dok_matrix((terms, topics))

    for i in xrange(topics):
        for j in xrange(terms):
            phi[j, i] = 1.0 / terms

    theta = dok_matrix((topics, docs))

    for i in xrange(docs):
        # theta[0, i] = 1.0
        for j in xrange(topics):
            theta[j, i] = 1.0 / topics

    return phi, theta


def sm(rows, columns):
    """ Create empty sparse matrix with giver shape """
    return dok_matrix((rows, columns))

def relative_frequencies_tdm(tdm_csc):
    freq_tdm = dok_matrix(tdm_csc.shape)
    print "converting to dok"
    tdm_dok = tdm_csc.todok()
    print "computing frequencies"

    for d in xrange(tdm_csc.shape[1]):
        denom = sum(tdm_csc[:, d].toarray(1)) + 0.0
        for w in tdm_csc[:, d].nonzero()[0]:
            freq_tdm[w, d] = (tdm_dok[w, d]) / (denom + 0.0001)
    return freq_tdm