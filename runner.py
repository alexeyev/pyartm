# !/usr/bin/python
# -*- coding: utf-8 -*-

from os import listdir
from os.path import isfile, join

from learner import *

# Choosing a directory with texts

path = "more"
onlyfiles = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
print onlyfiles

# Building a term-document matrix

words, tdm = build_tdm(onlyfiles[:4], min_df=0.25, max_df=0.70)

print "TDM built, starting EM..."

lrnr = EMStaticRegLearner(iter_number=40)
wt, td = lrnr.learn(tdm, 5)

print "It is done."

print (wt * td).todense()
print relative_frequencies_tdm(tdm).todense()
