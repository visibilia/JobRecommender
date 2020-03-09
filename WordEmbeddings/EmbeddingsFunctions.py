#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 15:41:09 2020

@author: fabiana
"""
from nltk.tokenize import sent_tokenize,word_tokenize
from DataPreparation.Preprocessing import preprocess_text_phrases

# preprocessing data for training word embedding model
def getSentences(data):
    
    sentences = [sent_tokenize(x.lower()) for x in data["descricao"]] 
    sentences += data["titulo"].str.lower().tolist()
    #print(len(sentences))
    
    return(sentences)

# getting the words of the sentences  
def getWords(sentences):
     
    word_sentences = [[word_tokenize(w) for w in s] for s in sentences]
    #len(word_sentences)
    final_word_sentences = preprocess_text_phrases(word_sentences)    
    
    return(final_word_sentences)