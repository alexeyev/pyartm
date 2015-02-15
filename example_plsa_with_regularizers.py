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

    path = "corpora/test"
    onlyfiles = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
    print "files:", ", ".join(onlyfiles)

    # Building a term-document matrix

    words, tdm = build_tdm_from_files(onlyfiles, min_df=0.25, max_df=0.7)

    # texts = []
    # for line in open("corpora/chgk.dataset/chgk.text.csv"):
    # text = line.split(";")[1]
    # print text.strip()
    #     texts.append(text)
    #
    # words, tdm = build_tdm_from_texts(texts, min_df=0.25, max_df=1.0)

    print "TDM built, starting EM..."

    learner = DumbEMStaticRegLearner(iter_number=50)
    wt, td = learner.learn(tdm, topics_number=5)

    print "It is done."

    result = (wt * td).todense()

    # for word, row in zip(result, words):
    # print word, row

    print "\nword -> topic\n"

    # for word, row in zip(wt.todense(), words):
    #     print word, row

    for i in xrange(wt.shape[1]):
        col = wt.getcol(i)
        colarray = col.transpose().toarray()[0]
        wordedcol = zip(words, colarray)
        print "\ntopic", i, "|- ", " ".join(
            map(lambda x: x[0], sorted(wordedcol, reverse=True, key=lambda x: x[1])[:20]))

    print "\ntopic -> document\n"

    # for doc, filename in zip(td.transpose().todense(), onlyfiles):
    #     print doc, filename