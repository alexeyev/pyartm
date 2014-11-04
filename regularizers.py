# !/usr/bin/python
# -*- coding: utf-8 -*-

class Reg:

    def __init__(self, shape_wt, shape_td):
        pass

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