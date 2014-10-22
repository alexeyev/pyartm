#!/usr/bin/python
# -*- coding: utf-8 -*-

import nltk
import textmining
import numpy

stemmer = nltk.stem.porter.PorterStemmer()
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

def get_words(text):
    tokenized = tokenizer.tokenize(text)
    return [stemmer.stem(t).lower() for t in tokenized]

def norm(text):
    return " ".join(get_words(text))


def build_matrix(texts):
    """
    texts are filenames
    inefficient implementation
    """
    print texts
    tdm = textmining.TermDocumentMatrix()
    for text in texts:
        data = open(text, "r").read()
        tdm.add_doc(norm(data))
    rows_in_mem = [row for row in tdm.rows(cutoff=0)]
    rest = rows_in_mem[1:]
    headers = rows_in_mem[0]
    return headers, numpy.array([row for row in rest])


def em(topics, tdm):
    d = tdm.shape[0]
    w = tdm.shape[1]
    wt = numpy.array([1.0 / w] * (w * topics)).reshape(w, topics)
    return wt

print build_matrix(["cartoon.txt", "book.txt"])

print em(10, build_matrix(["cartoon.txt", "book.txt"])[1])