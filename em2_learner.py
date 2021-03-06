# !/usr/bin/python
# -*- coding: utf-8 -*-

from learner import *


class DumbEMStaticRegLearner(Learner):
    def __init__(self, iter_number=50, regularizers=[], reg_coefficients=[]):
        self.iter = iter_number
        self.regs = regularizers
        self.reg_coefficients = reg_coefficients

    def learn(self, wdm, topics_number):
        return self.__em(wdm, topics_number, self.iter)

    def __em(self, tdm_csc, topics, iterations):
        """
            :param tdm_csc: term-document sparse matrix; words are rows, docs are columns
            :param topics: suggested number of topics
            :param iterations:
            :return:
        """

        # print "computing relative TDM frequencies"
        # freq_tdm = relative_frequencies_tdm(tdm_csc)

        print "converting tdm to DOK matrix"
        print "tdm shape = ", tdm_csc.shape
        tdm = tdm_csc.todok()

        words, docs = tdm_csc.shape
        print "initializing helper matrices"
        phi, theta = init_matrices(words, docs, topics)

        i = 0
        normm = 10.0

        print "iterations..."
        while i < iterations and normm > 0.000001:

            phi_old = dok_matrix(phi)
            theta_old = dok_matrix(theta)

            print "---------------------------------"
            print "iteration #", i
            i += 1

            nwt, ntd, nd, nt = sm(words, topics), sm(topics, docs), sm(docs, 1), sm(topics, 1)
            # 3d matrix of p(tdw); p[topic][doc, word]
            # p_tdw = array(array(dok_matrix((docs, words))).repeat(topics))

            print "E-step"

            for d in xrange(docs):
                for w in tdm_csc[:, d].nonzero()[0]:
                    # print w, d, "|where ", w," is non zero in tdm in this doc"
                    phi_theta_dw = 0.0
                    for t in xrange(topics):
                        phi_theta_dw += phi[w, t] * theta[t, d]
                    for t in xrange(topics):
                        if phi_theta_dw > 0:
                            exp_ndwt = (phi[w, t] * theta[t, d]) / phi_theta_dw * tdm[w, d]
                            nwt[w, t] += exp_ndwt
                            ntd[t, d] += exp_ndwt

            print "M-step"

            phisums = sm(topics, 1)

            for w in xrange(words):
                for t in xrange(topics):
                    reg_eval_sum = 0.0
                    for r_id in xrange(len(self.regs)):
                        reg_eval_sum += phi[w, t] \
                                        * self.reg_coefficients[r_id] \
                                        * self.regs[r_id].df(phi, theta)[0][w, t]
                    phi[w, t] = pos(nwt[w, t] + reg_eval_sum)
                    phisums[t, 0] += phi[w, t]

            for w in xrange(words):
                for t in xrange(topics):
                    if phisums[t, 0] != 0:
                        phi[w, t] /= phisums[t, 0]

            thetasums = sm(docs, 1)

            for d in xrange(docs):
                for t in xrange(topics):
                    reg_eval_sum = 0.0
                    for r_id in xrange(len(self.regs)):
                        reg_eval_sum += theta[t, d] \
                                        * self.reg_coefficients[r_id] \
                                        * self.regs[r_id].df(phi, theta)[1][t, d]
                    theta[t, d] = pos(ntd[t, d] + reg_eval_sum)
                    thetasums[d, 0] += theta[t, d]

            for d in xrange(docs):
                if thetasums[d, 0] != 0:
                    for t in xrange(topics):
                        theta[t, d] /= thetasums[d, 0]

            normm = norm((phi - phi_old).toarray())
            print "phi diff = ", normm
            print "theta diff = ", norm((theta - theta_old).toarray())
            # print "recovery norm = ", norm((freq_tdm - (phi * theta)).toarray())

        return phi, theta