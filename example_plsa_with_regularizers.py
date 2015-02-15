# !/usr/bin/python
# -*- coding: utf-8 -*-

from os import listdir
from os.path import isfile, join
from em2_learner import DumbEMStaticRegLearner

from learner import *
import scipy
from regularizers import ZeroRegularizer, LDARegularizer

if __name__ == '__main__':

    # Choosing a directory with texts

    # path = "corpora/test"
    path = "corpora/chgk.dataset/chgk.text.csv"

    if not "dataset" in path:
        onlyfiles = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
        print "files:", ", ".join(onlyfiles)
        words, tdm = build_tdm_from_files(onlyfiles, min_df=0.0005, max_df=0.7)
    else:
        texts = []
        count = 0
        for line in open(path):
            count += 1
            if count % 10000 == 0:
                print count, "lines read"
            texts.append(line.split(";")[1])
        words, tdm = build_tdm_from_texts(texts[:100], min_df=0.05, max_df=0.9)

    print "TDM built, starting EM..."

    learner = DumbEMStaticRegLearner(iter_number=500)
    wt, td = learner.learn(tdm, topics_number=10)

    print "It is done."

    result = (wt * td).todense()

    # for word, row in zip(result, words):
    # print word, row

    print "\nword -> topic\n"

    # for word, row in zip(wt.todense(), words):
    # print word, row

    for i in xrange(wt.shape[1]):
        col = wt.getcol(i)
        colarray = col.transpose().toarray()[0]
        wordedcol = zip(words, colarray)
        print "topic", i, "|- ", " ".join(
            map(lambda x: x[0], sorted(wordedcol, reverse=True, key=lambda x: x[1])[:20]))

    print "\ntopic -> document\n"

    # for doc, filename in zip(td.transpose().todense(), onlyfiles):
    # print doc, filename