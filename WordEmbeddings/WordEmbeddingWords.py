#!/usr/bin/env python
# coding: utf-8

# importing references
from gensim import models as gsm
import pandas as pd
from nltk.tokenize import sent_tokenize,word_tokenize
from DataPreparation.Preprocessing import preprocess_text
from time import time


class WordEmbWords(): 

    def __init__(self, dataFile, out):
        self.dataFile = dataFile
        self.out = out
    
    def getWordEmbModel(self):
        
        print("Feature representation step - Word Embeddings-Words \n")
        
        # loading data
        vagas_ti = pd.read_csv(self.dataFile)

        # preprocessing data for training word embedding model
        sentences = [sent_tokenize(x.lower()) for x in vagas_ti["descricao"]] 
        sentences += vagas_ti["titulo"].str.lower().tolist()

        word_sentences = [[word_tokenize(w) for w in s] for s in sentences]

        # preprocessing word-segmented sentences
        final_word_sentences = preprocess_text(word_sentences)
        
        start = time()
        model_skill = gsm.Word2Vec(final_word_sentences, min_count=2,workers=3,sg=1,size=200)
        model_skill.save(self.out+"wordEmbeddings/ti_skill_w2v_skg_200.model")
        print("It took: %.4f"%(time()-start))
        
        start = time()
        model_skill = gsm.Word2Vec(final_word_sentences, min_count=2,workers=3,sg=0,size=200)
        model_skill.save(self.out+"wordEmbeddings/ti_skill_w2v_cbow_200.model")
        print("It took: %.4f"%(time()-start))
        
        print("\n Feature representation - Word Embeddings-Words done!")





# Working with phrases
#bigram_transformer = gsm.phrases.Phrases(final_word_sentences)
#print(list(bigram_transformer[sentences]))
#model = Word2Vec(bigram_transformer[sentences], size=100, ...)
#start = time()
#phrases = gsm.phrases.Phrases(final_word_sentences)
#bigram = gsm.phrases.Phraser(phrases)
#model_skill2 = gsm.Word2Vec(phrases[final_word_sentences], min_count=2,workers=3,sg=1)
#model_skill2.save("ti_skill_phrases.model")
#model = Word2Vec.load(fname)  # you can continue training with the loaded model!
#print("It took: %.4f"%(time()-start))


