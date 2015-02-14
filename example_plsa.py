# !/usr/bin/python
# -*- coding: utf-8 -*-

from os import listdir
from os.path import isfile, join

from learner import *
import scipy

if __name__ == '__main__':

    # Choosing a directory with texts

    path = "corpus/small_test"
    onlyfiles = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
    print "files:", ", ".join(onlyfiles)

    # Building a term-document matrix

    words, tdm = build_tdm(onlyfiles, min_df=0.25, max_df=1.0)

    print tdm.todense()

    print "TDM built, starting EM..."

    learner = EMStaticRegLearner(iter_number=100)
    wt, td = learner.learn(tdm, topics_number=2)

    print "It is done."

    result = (wt * td).todense()

    for word, row in zip(result, words):
        print word, row

    print
    print relative_frequencies_tdm(wt * td).todense()

    print "\nword -> topic\n"

    for word, row in zip(wt.todense(), words):
        print word, row

    for i in xrange(wt.shape[1]):
        col = wt.getcol(i)
        colarray = col.transpose().toarray()[0]
        wordedcol = zip(words, colarray)
        print "topic", i, " | ".join(
            map(lambda x: x[0],
                filter(lambda x: x[1] > 0.008, sorted(wordedcol, reverse=True, key=lambda x: x[1]))))

    print "\ntopic -> document\n"

    for doc, filename in zip(td.transpose().todense(), onlyfiles):
        print doc, filename