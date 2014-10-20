#!/usr/bin/python
# -*- coding: utf-8 -*-

import nltk
import textmining

stemmer = nltk.stem.porter.PorterStemmer()
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

def get_words(text):
    tokenized = tokenizer.tokenize(text)
    return [stemmer.stem(t).lower() for t in tokenized]

text1 = """Alice was published in 1865, three years after Charles Lutwidge Dodgson and the Reverend Robinson Duckworth rowed in a boat, on 4 July 1862[4] (this popular date of the "golden afternoon"[5] might be a confusion or even another Alice-tale, for that particular day was cool, cloudy and rainy[6]), up the Isis with the three young daughters of Henry Liddell (the Vice-Chancellor of Oxford University and Dean of Christ Church): Lorina Charlotte Liddell (aged 13, born 1849) ("Prima" in the book's prefatory verse); Alice Pleasance Liddell (aged 10, born 1852) ("Secunda" in the prefatory verse); Edith Mary Liddell (aged 8, born 1853) ("Tertia" in the prefatory verse)."""

text2 = """The Queen of Hearts She made some tarts,All on a summer's day;The Knave of Hearts He stole those tarts,And took them clean away.The King of Hearts Called for the tarts,And beat the knave full sore;The Knave of Hearts Brought back the tarts,And vowed he'd steal no more. """

text3 = """The Queen of Hearts She made some tarts,All on a summer's day;The Knave of Hearts He stole those tarts,And took them clean away.The King of Hearts Called for the tarts,And beat the knave full sore;The Knave of Hearts Brought back the tarts,And vowed he'd steal no more. """

#print get_words(text1)
#print get_words(text2)

tdm = textmining.TermDocumentMatrix()
tdm.add_doc(text1)
tdm.add_doc(text2)
tdm.add_doc(text3)

for row in tdm.rows(cutoff=1):
    print row
