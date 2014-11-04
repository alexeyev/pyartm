# !/usr/bin/python
# -*- coding: utf-8 -*-

from numpy.linalg import norm

from matrices_utils import *


class Learner:
    def learn(self, wdm, topics_number):
        raise NotImplementedError("Please override `learn` method")


class EMStaticRegLearner(Learner):
    def __init__(self, iter_number=50, regularizers=[], reg_coefficients=[]):
        self.iter = iter_number
        self.regs = regularizers
        self.reg_coefficients = reg_coefficients

    def learn(self, wdm, topics_number):
        print "PLEASE NOTE: REGULARIZERS ARE NOT SUPPORTED YET; ENJOY PLAIN P-LSA"
        return self.__em(wdm, topics_number, self.iter)

    def __em(self, tdm_csc, topics, iterations):
        """
            :param tdm_csc: term-document sparse matrix; words are rows, docs are columns
            :param topics: suggested number of topics
            :param iterations:
            :return:
        """
        print "computing relative TDM frequencies"
        freq_tdm = relative_frequencies_tdm(tdm_csc)

        print "converting tdm to DOK matrix"
        tdm = tdm_csc.todok()

        words, docs = tdm_csc.shape
        print "initializing helper matrices"
        phi, theta = init_matrices(words, docs, topics)

        i = 0
        norm_val = 2.0

        while i < iterations:  # and norm_val > 0.01:

            print "---------------------------------"
            print "iteration #", i, "norm = ", norm_val
            i += 1

            nwt, ntd, nd = sm(words, topics), sm(topics, docs), sm(docs, 1)

            print "computing expectations"
            print "docs",

            for d in xrange(docs):
                if d % 50 == 0:
                    print d,
                for w in tdm_csc[:, d].nonzero()[0]:
                    phi_theta_dw = 0.0
                    for t in xrange(topics):
                        phi_theta_dw += phi[w, t] * theta[t, d]
                    for t in xrange(topics):
                        if phi[w, t] != 0 and theta[t, d] != 0:
                            expe = (phi[w, t] * theta[t, d]) / (phi_theta_dw + 0.0001) * tdm[w, d]
                            nwt[w, t] += expe
                            ntd[t, d] += expe
                            nd[d, 0] += expe

            print
            print "reestimating phi and theta"

            nwt = nwt.tocsc()

            for t in xrange(topics):
                print "topic", t, " # of words = ", words
                nt_t = sum(nwt.getcol(t).toarray(1)) + 0.0001
                for w in xrange(words):
                    phi[w, t] = nwt[w, t] / nt_t
                for d in xrange(docs):
                    theta[t, d] = ntd[t, d] / (nd[d, 0] + 0.0001)

            print "computing norm"
            # todo: think a little
            norm_val = norm(((phi.tocsr() * theta.tocsc()) - freq_tdm.tocsr()).toarray(2), 1)

        return phi, theta