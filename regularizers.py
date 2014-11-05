# !/usr/bin/python
# -*- coding: utf-8 -*-

from scipy.sparse import *


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
        return dok_matrix(wt.shape), dok_matrix(td.shape)

    def df(self, wt, td):
        return dok_matrix(wt.shape), dok_matrix(td.shape)