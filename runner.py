# !/usr/bin/python
# -*- coding: utf-8 -*-

from os import listdir
from os.path import isfile, join

from learner import *
import scipy

# Choosing a directory with texts

# path = "corpus/test"
# onlyfiles = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
# print onlyfiles
#
# # Building a term-document matrix
#
# words, tdm = build_tdm(onlyfiles[:6], min_df=0.0, max_df=1.0)
#
# print "TDM built, starting EM..."
#
# lrnr = EMStaticRegLearner(iter_number=5)
# wt, td = lrnr.learn(tdm, 3)
#
# print "It is done."
#
# result = (wt * td).todense()
# # print result
#
# for word, row in zip(words, result):
# print row, word
#
# for word, row in zip(words, relative_frequencies_tdm(tdm).todense()):
#     print row, word


if __name__ == '__main__':

    # Choosing a directory with texts

    path = "corpus/test"
    onlyfiles = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
    print "files:", ", ".join(onlyfiles)

    # Building a term-document matrix

    words, tdm = build_tdm(onlyfiles, min_df=0.25, max_df=0.70)

    print tdm.todense()

    print "TDM built, starting EM..."

    learner = EMStaticRegLearner(iter_number=200)
    # DumbEMStaticRegLearner(iter_number=2, regularizers=[ZeroRegularizer()], reg_coefficients=[0.33])
    wt, td = learner.learn(tdm, 3)

    print "It is done."

    result = (wt * td).todense()

    for word, row in zip(result, words):
        print word, row

    print "word -> topic"

    for word, row in zip(wt.todense(), words):
        print word, row

    print "topic -> document"

    for doc, filename in zip(td.transpose().todense(),  onlyfiles):
        print doc, filename