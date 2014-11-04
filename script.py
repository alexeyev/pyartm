# !/usr/bin/python
# -*- coding: utf-8 -*-

import nltk
from scipy import *
from scipy.sparse import *
import sklearn.feature_extraction as fe
from numpy.linalg import norm

"""

@tdm -- W_[wd] -- term-document matrix; rows are words, columns are documents


"""


def build_tdm(filenames):
    texts = [open(fn, "r+").read() for fn in filenames]
    stopwords = nltk.corpus.stopwords.words('english')
    vectorizer = fe.text.CountVectorizer(min_df=0.1, ngram_range=(1, 1), max_df=0.30, stop_words=stopwords)

    X_words = vectorizer.fit_transform(texts).transpose()

    for i in xrange(len(vectorizer.get_feature_names())):
        print vectorizer.get_feature_names()[i], "\t\t\t",
        print X_words[i, :].todense()
    return vectorizer.get_feature_names(), X_words


def init_matrices(terms, docs, topics):
    """ phi_w,t = p(w | t), theta_t,d = p(t | d) """
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
    return dok_matrix((rows, columns))


def relative_frequencies_tdm(tdm):
    freq_tdm = dok_matrix(tdm.shape)

    for d in xrange(tdm.shape[1]):
        denom = sum(tdm[:, d].todense()) + 0.0
        for w in xrange(tdm.shape[0]):
            freq_tdm[w, d] = (tdm[w, d] + 0.0001) / (denom + 0.0001)
    return freq_tdm


def em(tdm, topics, iterations):
    freq_tdm = relative_frequencies_tdm(tdm)
    tdm_csc = tdm.tocsc()

    words, docs = tdm.shape
    phi, theta = init_matrices(words, docs, topics)

    i = 0
    norm_val = 2.0
    while i < iterations and norm_val > 0.01:

        print "iteration #", i, "norm = ", norm_val
        i += 1

        nwt, ntd, nd = sm(words, topics), sm(topics, docs), sm(docs, 1)

        print "computing expectations"

        # phi_csr = phi.tocsr()
        # theta_csc = theta.tocsc()

        for d in xrange(docs):
            print "doc", d
            for w in tdm_csc[:, d].nonzero()[0]:
                # phi_theta_dw = phi_csr[w].dot(theta_csc[:, d])[0, 0]
                phi_theta_dw = 0.0
                for t in xrange(topics):
                    phi_theta_dw += phi[w, t] * theta[t, d]

                for t in xrange(topics):
                    if phi[w, t] != 0 and theta[t, d] != 0:
                        expe = (phi[w, t] * theta[t, d]) / (phi_theta_dw + 0.0001) * tdm[w, d]
                        nwt[w, t] += expe
                        ntd[t, d] += expe
                        # nt[t, 0] += exp
                        nd[d, 0] += expe

        print "reestimating phi and theta"

        ntd = ntd.tocsc()
        nwt = nwt.tocsc()

        for t in xrange(topics):
            print "topic", t, " # of words = ", words
            nt_t = sum(nwt.getcol(t)[0, 0]) + 0.0001
            print nt_t
            for w in xrange(words):
                phi[w, t] = nwt[w, t] / nt_t
            for d in xrange(docs):
                theta[t, d] = ntd[t, d] / (nd[d, 0] + 0.0001)  # (sum(ntd.getcol(d).toarray(1)) + 0.0001)  #

        print "computing norm"
        norm_val = norm(((phi.tocsr() * theta.tocsr()) - freq_tdm.tocsr()).toarray(2), 1)

    return phi, theta


from os import listdir
from os.path import isfile, join

path = "more"
onlyfiles = [join(path, f) for f in listdir(path) if isfile(join(path, f))]

print onlyfiles

words, tdm = build_tdm(onlyfiles[:10])

wt, td = em(tdm, 2, 100)
print (wt * td).todense()
