# !/usr/bin/python
# -*- coding: utf-8 -*-

from scipy.sparse import *
from math import *


class Reg:
    def f(self, wt, td):
        """
        Provides regularizer value given two matrices
        :param wt: word-to-topic matrix
        :param td: topic-to-document matrix
        """
        raise NotImplementedError()

    def df(self, wt, td):
        """
        Provides regularizer first derivative values given two matrices
        :param wt: word-to-topic matrix
        :param td: topic-to-document matrix
        :return:
        """
        raise NotImplementedError()


class ZeroRegularizer(Reg):
    def f(self, wt, td):
        return 0

    def df(self, wt, td):
        return dok_matrix(wt.shape), dok_matrix(td.shape)


class LDARegularizer(Reg):
    def __init__(self, wt_bias, wt_distr, td_bias, td_distr):
        """ *_bias -- int; *_distr -- python arrays  """
        self.wt_bias, self.wt_distr = wt_bias, wt_distr
        self.td_bias, self.td_distr = td_bias, td_distr
        self.topics = len(td_distr)
        self.words = len(wt_distr)

    def f(self, wt, td):
        # todo: efficient implementation
        summer = 0.0
        docs = td.shape[1]
        for t in xrange(self.topics):
            wt_summer = 0.0
            for w in xrange(self.words):
                wt_summer += math.log(wt[w, t]) * self.wt_distr[w]
            wt_summer *= self.wt_bias
            td_summer = 0.0
            for d in xrange(docs):
                td_summer += math.log(td[t, d]) * self.td_distr[t]
            td_summer *= self.td_bias
            summer += wt_summer + td_summer
        return summer

    def df(self, wt, td):
        """
            dR/dPhi_wt = wt_bias * wt_distr[w] / Phi_wt
            dR/dTheta_td = ts_bias * ts_distr[t] / Theta_wtd
        """
        dphi, dtheta = dok_matrix(wt.shape), dok_matrix(td.shape)
        docs = td.shape[1]
        # todo: efficient implementation
        for t in xrange(self.topics):
            for w in xrange(self.words):
                dphi[w, t] = self.wt_bias * self.wt_distr[w] / (wt[w, t] + 0.00001)
            for d in xrange(docs):
                dtheta[t, d] = self.td_bias * self.td_distr[t] / (td[t, d] + 0.00001)
        return dphi, dtheta


class EmbeddingsRegularizer(Reg):

    def __init__(self, word2vector_dict):
        pass

    def f(self, wt, td):
        raise NotImplementedError

    def df(self, wt, td):
        raise NotImplementedError