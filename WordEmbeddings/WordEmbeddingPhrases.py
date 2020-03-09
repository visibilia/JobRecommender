# -*- coding: utf-8 -*-

from gensim import models as gsm
import pandas as pd
from time import time
import pickle
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from WordEmbeddings.EmbeddingsFunctions import getWords, getSentences

class WordEmbPhrases():
    
    def __init__(self, dataFile, out):
        self.dataFile = dataFile
        self.out = out

    def getWordEmbModel_ngram(self, model, word_sentences):

        # Training model of word embedding with phrases
        bigram = gsm.phrases.Phrases(word_sentences)
        bigram = gsm.phrases.Phraser(bigram) 
        trigram = gsm.phrases.Phrases(bigram[word_sentences])
        trigram = gsm.phrases.Phraser(trigram)
    
        # Saving preprocessed cvs
        fb = open(self.out+"wordEmbeddings/vagas_cv.bigram",'wb')
        ft = open(self.out+"wordEmbeddings/vagas_cv.trigram",'wb')
        pickle.dump(bigram,fb)
        pickle.dump(trigram,ft)
        #print("Phrasers saved!")
        
        if(model == "skipgram"):
            start = time()
            model_skill = gsm.Word2Vec(trigram[bigram[word_sentences]], min_count=2, workers=3, sg=1, size=200)
            model_skill.save(self.out+"wordEmbeddings/ti_skill_phrases_skg.model")
            print("It took: %.4f"%(time()-start))
        
        else: # for cbow
            start = time()
            model_skill = gsm.Word2Vec(trigram[bigram[word_sentences]], min_count=2, workers=3, sg=0, size=200)
            model_skill.save(self.out+"wordEmbeddings/ti_skill_phrases_cbow.model")
            print("It took: %.4f"%(time()-start))
            

    def main(self):
        
        print("Feature representation step - Word Embeddings-Phrases")
        
        vagas_ti = pd.read_csv(self.dataFile, encoding="utf8")
        #print(vagas_ti.shape)
        
        sentences = getSentences(vagas_ti)
        final_word_sentences = getWords(sentences)
        self.getWordEmbModel_ngram("skipgram", final_word_sentences)
        self.getWordEmbModel_ngram("cbow", final_word_sentences)
        
        print("Feature representation - Word Embeddings-Phrases done!")




# Loading bigram and trigrams
#bigram = pickle.load(open("WordEmbeddings/vagas_cv.bigram","rb"))
#trigram = pickle.load(open("WordEmbeddings/vagas_cv.trigram","rb"))
#sent2 = ['ser',
#  'referência',
#  'área',
#  'quantum',
#  'tecnologia',
#  'java',
#  'diferencial',
#  'angular',
#  'js',
#  'ensino',
#  'superior',
#  'completo',
#  'tecnologia',
#  'informação',
#  'área',
#  'afins',
#  'desejável',
#  'conhecimento',
#  'json',
#  'xml',
#  'sql',
#  'j2me']
#
#print(trigram[bigram[sent2]])
#
#
#
#sent2 = [['ser',
#  'referência',
#  'área',
#  'quantum',
#  'tecnologia',
#  'java',
#  'diferencial',
#  'angular',
#  'js'],['ensino',
#  'superior',
#  'completo',
#  'tecnologia',
#  'informação',
#  'área',
#  'afins',
#  'desejável',
#  'conhecimento',
#  'json',
#  'xml',
#  'sql',
#  'j2me']]
#print(list(trigram[bigram[sent2]]))

