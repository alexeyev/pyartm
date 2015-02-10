# !/usr/bin/python
# -*- coding: utf-8 -*-

from os import listdir
from os.path import isfile, join

from learner import *
from regularizers import ZeroRegularizer, LDARegularizer


class DumbEMStaticRegLearner(Learner):
    # todo: rewrite and make it work
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
        tdm = tdm_csc.todok()

        words, docs = tdm_csc.shape
        print "initializing helper matrices"
        phi, theta = init_matrices(words, docs, topics)

        print "before shit\n", (phi.tocsr() * theta.tocsc()).todense()
        print "phi\n", phi.todense()
        print "theta\n", theta.todense()

        i = 0

        while i < iterations:  # and norm_val > 0.01:

            print "---------------------------------"
            print "iteration #", i  # , "norm = ", norm_val
            i += 1

            nwt, ntd, nd = sm(words, topics), sm(topics, docs), sm(docs, 1)
            # 3d matrix of p(tdw); p[topic][doc, word]
            p_tdw = array(array(dok_matrix((docs, words))).repeat(topics))

            print "E-step"

            # todo: speedup
            for d in xrange(docs):
                for w in xrange(words):
                    denom = 0.0
                    for t in xrange(topics):
                        denom += phi[w, t] * theta[t, d]
                    # print denom
                    for t in xrange(topics):
                        p_tdw[t][d, w] = phi[w, t] * theta[t, d] / denom

            print "M-step"

            # n_wt, n_td
            # todo: speedup
            for w in xrange(words):
                for t in xrange(topics):
                    for d in xrange(docs):
                        term = tdm[w, d] * p_tdw[t][d, w]
                        nwt[w, t] += term
                        ntd[t, d] += term

            for w in xrange(words):
                for t in xrange(topics):
                    reg_eval_sum = 0.0
                    for r_id in xrange(len(self.regs)):
                        # print self.regs
                        # print self.reg_coefficients
                        # print r_id
                        # self.reg_coefficients[r_id]
                        # self.regs[r_id]
                        # self.regs[r_id].df(phi, theta)[0][w, t]
                        reg_eval_sum += phi[w, t] * self.reg_coefficients[r_id] * self.regs[r_id].df(phi, theta)[0][
                            w, t]
                    phi[w, t] = pos(nwt[w, t] + reg_eval_sum)

            # phi_csc = phi.tocsc()
            for t in xrange(topics):
                denom = 0.0
                for w in xrange(words):
                    denom += phi[w, t]
                for w in xrange(words):
                    # print phi[w, t], "/", denom
                    phi[w, t] /= denom
                    # print phi[w, t]

            for d in xrange(docs):
                for t in xrange(topics):
                    reg_eval_sum = 0.0
                    for r_id in xrange(len(self.regs)):
                        reg_eval_sum += theta[t, d] * self.reg_coefficients[r_id] * self.regs[r_id].df(phi, theta)[1][
                            t, d]
                    theta[t, d] = pos(ntd[t, d] + reg_eval_sum)

            for d in xrange(docs):
                denom = 0.0
                for t in xrange(topics):
                    denom += theta[t, d]
                    print theta[t, d],
                print
                print denom
                for t in xrange(topics):
                    theta[t, d] /= denom

                    # print "computing norm"
                    # norm_val = norm(((phi.tocsr() * theta.tocsc()) - freq_tdm.tocsr()).toarray(2), 2)

        return phi, theta