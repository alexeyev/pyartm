# !/usr/bin/python
# -*- coding: utf-8 -*-

from os import listdir
from os.path import isfile, join
from em2_learner import DumbEMStaticRegLearner

from learner import *
import scipy
from regularizers import ZeroRegularizer

if __name__ == '__main__':

    # Choosing a directory with texts

    path = "corpus/chgk.dataset"
    onlyfiles = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
    print "files:", ", ".join(onlyfiles)

    # Building a term-document matrix

    words, tdm = build_tdm(onlyfiles, min_df=0.25, max_df=1.0)

    print tdm.todense()

    print "TDM built, starting EM..."

    learner = DumbEMStaticRegLearner(iter_number=500, regularizers=[],
                                     reg_coefficients=[])
    wt, td = learner.learn(tdm, topics_number=12)

    print "It is done."

    result = (wt * td).todense()

    # for word, row in zip(result, words):
    #     print word, row

    print "\nword -> topic\n"

    # for word, row in zip(wt.todense(), words):
    #     print word, row

    for i in xrange(wt.shape[1]):
        col = wt.getcol(i)
        colarray = col.transpose().toarray()[0]
        wordedcol = zip(words, colarray)
        print "topic", i, "|", ", ".join(
            map(lambda x: x[0],
                sorted(wordedcol, reverse=True, key=lambda x: x[1])[:20]))

    print "\ntopic -> document\n"

    # for doc, filename in zip(td.transpose().todense(), onlyfiles):
    #     print doc, filename