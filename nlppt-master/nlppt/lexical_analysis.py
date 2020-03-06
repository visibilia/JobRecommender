from nltk import sent_tokenize
from nltk import word_tokenize
from nltk.stem import RSLPStemmer

stemmer = RSLPStemmer()


def sentence_split(text):
    sentences = sent_tokenize(text, language='portuguese')
    return sentences


def tokenize(text):
    tokens = word_tokenize(text, language='portuguese')
    return tokens


def stem(token):
    return stemmer.stem(token)
