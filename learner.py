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
        print "computing relative TDM frequencies"
        freq_tdm = relative_frequencies_tdm(tdm_csc)

        print "converting tdm to DOK matrix"
        tdm = tdm_csc.todok()

        words, docs = tdm_csc.shape
        print "initializing helper matrices"
        phi, theta = init_matrices(words, docs, topics_number)

        i = 0
        norm_val = 2.0

        print "before shit\n", (phi.tocsr() * theta.tocsc()).todense()

        while i < iterations:  # and norm_val > 0.01:

            print "---------------------------------"
            print "iteration #", i, "norm = ", norm_val
            i += 1

            nwt, ntd, nd = sm(words, topics_number), sm(topics_number, docs), sm(docs, 1)

            print "computing expectations"
            print "docs",

            for d in xrange(docs):
                # if d % 50 == 0:
                #     print d,
                for w in tdm_csc[:, d].nonzero()[0]:
                    phi_theta_dw = 0.0
                    for t in xrange(topics_number):
                        phi_theta_dw += phi[w, t] * theta[t, d]
                    for t in xrange(topics_number):
                        exp_ndwt = (phi[w, t] * theta[t, d]) / (phi_theta_dw + 0.0001) * tdm[w, d]
                        nwt[w, t] += exp_ndwt
                        ntd[t, d] += exp_ndwt
                        nd[d, 0] += exp_ndwt

            print
            print "reestimating phi and theta"

            nwt = nwt.tocsc()

            for t in xrange(topics_number):
                print "topic", t
                nt_t = sum(nwt.getcol(t).toarray(1)) + 0.0001
                print "nt_t", nt_t
                for w in xrange(words):
                    phi[w, t] = nwt[w, t] / nt_t
                for d in xrange(docs):
                    theta[t, d] = ntd[t, d] / (nd[d, 0] + 0.0001)
            print "theta", theta.toarray(1)

            print "norm =                                    " , norm((phi.tocsr() * theta.tocsc() - freq_tdm.tocsr()).toarray())
            print "*\n", (phi.tocsr() * theta.tocsc()).todense()
            print "freq tdm\n", freq_tdm.todense()
            # norm_val = norm(phi.tocsr() * theta.tocsc() - freq_tdm.tocsr(), 1)

        return phi, theta