# !/usr/bin/python
# -*- coding: utf-8 -*-

import nltk
import numpy
from scipy import *
from scipy.sparse import *
import sklearn.feature_extraction as fe


def build_tdm(filenames):
    texts = [open(fn, "r+").read() for fn in filenames]
    stopwords = nltk.corpus.stopwords.words('english')
    vectorizer = fe.text.CountVectorizer(min_df=0.2, ngram_range=(1, 2), max_df=1, stop_words=stopwords)

    # строки -- документы, столбцы -- слова
    X_words = vectorizer.fit_transform(texts)
    return vectorizer.get_feature_names(), X_words


def init_matrices(terms, docs, topics):
    """ phi_w,t = p(w | t), theta_t,d = p(t | d) """
    phi = lil_matrix((terms, topics))

    for i in xrange(min(terms, topics)):
        phi[i, i] = 1.0

    theta = csc_matrix((topics, docs))

    for i in xrange(min(docs, topics)):
        phi[i, i] = 1.0

    return phi, theta


def sm(rows, columns):
    return lil_matrix((rows, columns))


def em(tdm, topics, iterations):
    docs, words = tdm.shape
    phi, theta = init_matrices(words, docs, topics)

    for i in xrange(iterations):

        print i, "iteration"

        nwt, ntd, nt, nd = sm(words, topics), sm(topics, docs), sm(topics, 1), sm(docs, 1)
        print "matrices set"

        for d in xrange(docs):
            print "doc", d
            for w in xrange(words):
                if w % 500 == 0:
                    print w,
                phi_theta_dw = 0.0

                for t in xrange(topics):
                    phi_theta_dw += phi[w, t] * theta[t, d]

                for t in xrange(topics):
                    ptwd = phi[w, t] * theta[t, d] / (phi_theta_dw + 0.001)
                    exp = ptwd * tdm[d, w]
                    nwt[w, t] += exp
                    ntd[t, d] += exp
                    nt[t, 0] += exp
                    nd[d, 0] += exp
            print

        for t in xrange(topics):
            print "topic", t
            for w in xrange(words):
                phi[w, t] = nwt[w, t] / (nt[t, 0] + 0.0001)
            for d in xrange(docs):
                theta[t, d] = ntd[t, d] / (nd[d, 0] + 0.0001)
    return phi, theta

from os import listdir
from os.path import isfile, join
path = "corpus"
onlyfiles = [join(path,f) for f in listdir(path) if isfile(join(path,f))]

print onlyfiles

tdm = build_tdm(onlyfiles[:4])[1]

print tdm
print tdm.shape

wt, td = em(tdm, 10, 2)

print wt.todense()
