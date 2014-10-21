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
    tdm = textmining.TermDocumentMatrix()
    for text in texts:
        data = open(text, "r").read()
        tdm.add_doc(norm(data))
    return numpy.array([row for row in tdm.rows(cutoff=0)][1:])


def em(topics, tdm):
    return 0

print build_matrix(["cartoon.txt", "book.txt"])