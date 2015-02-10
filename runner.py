# !/usr/bin/python
# -*- coding: utf-8 -*-

from os import listdir
from os.path import isfile, join

from learner import *
import scipy

# Choosing a directory with texts

path = "corpus/smalltest"
onlyfiles = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
print onlyfiles

# Building a term-document matrix

words, tdm = build_tdm(onlyfiles[:6], min_df=0.0, max_df=1.0)

print "TDM built, starting EM..."

lrnr = EMStaticRegLearner(iter_number=5)
wt, td = lrnr.learn(tdm, 3)

print "It is done."

result = (wt * td).todense()
# print result

for word, row in zip(words, result):
    print row, word

for word, row in zip(words, relative_frequencies_tdm(tdm).todense()):
    print row, word
