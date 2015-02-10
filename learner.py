# !/usr/bin/python
# -*- coding: utf-8 -*-

import numpy
from numpy.linalg import norm

from matrices_utils import *


class Learner:
    def learn(self, wdm, topics_number):
        raise NotImplementedError("Please override `learn` method")


class EMStaticRegLearner(Learner):
    """
        EM learning without any regularizers
    """

    def __init__(self, iter_number=50):
        self.iter = iter_number

    def learn(self, wdm, topics_number):
        return self.__em(wdm, topics_number, self.iter)

    def __em(self, tdm_csc, topics_number, iterations):
        """
            :param tdm_csc: term-document sparse matrix; words are rows, docs are columns
            :param topics_number: suggested number of topics
            :param iterations:
            :return:
        """

        tdm = tdm_csc.todok()
        words, docs = tdm_csc.shape
        phi, theta = init_matrices(words, docs, topics_number)

        i = 0

        print "before shit\n", (phi.tocsr() * theta.tocsc()).todense()
        print "phi\n", phi.todense()
        print "theta\n", theta.todense()

        while i < iterations:

            print "iteration #", i,
            i += 1

            nwt, ntd, nd, nt = sm(words, topics_number), sm(topics_number, docs), sm(docs, 1), sm(topics_number, 1)

            print "E...",


            for d in xrange(docs):
                for w in tdm_csc[:, d].nonzero()[0]:
                    # print w, d, "|where ", w," is non zero in tdm in this doc"
                    phi_theta_dw = 0.0
                    for t in xrange(topics_number):
                        phi_theta_dw += phi[w, t] * theta[t, d]
                    for t in xrange(topics_number):
                        exp_ndwt = (phi[w, t] * theta[t, d]) / (phi_theta_dw + 0.0001) * tdm[w, d]
                        nwt[w, t] += exp_ndwt
                        ntd[t, d] += exp_ndwt
                        nd[d, 0] += exp_ndwt
                        nt[t, 0] += exp_ndwt

            print "M..."

            nwt = nwt.tocsc()

            for t in xrange(topics_number):
                for w in xrange(words):
                    phi[w, t] = nwt[w, t] / (nt[t, 0] + 0.001)
                for d in xrange(docs):
                    theta[t, d] = ntd[t, d] / (nd[d, 0] + 0.0001)

        return phi, theta