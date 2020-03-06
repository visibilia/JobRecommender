import os
os.environ['NLTK_DATA'] =  os.path.abspath('nlppt/data')
from .lexical_analysis import sentence_split, tokenize, stem
from .morphological_analysis import lemma, morf
