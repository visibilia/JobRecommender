#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
#import nlppt as nl
from tqdm import tqdm
from nltk.corpus import stopwords
from time import time
from nltk.tokenize import sent_tokenize,word_tokenize
import pickle

def preprocess_text_phrases(ws):
    """stopwords + alphanumeric + removing ending punctuation"""
    punctuation = ["/",",",".",";",":","(",")","-","[","]"]
    final_word_sentences = []
    for v in tqdm(ws):
        for s in v:
            s = [w for w in s if w not in punctuation]
            s = [w for w in s if not(re.match(r"\d",w))] 
            s = [w for w in s if w not in stopwords.words("portuguese")]
            final_word_sentences.append(s)
    final_word_sentences = [[w[:-1] if w[-1] in punctuation else w for w in s] for s in final_word_sentences]
    return final_word_sentences


def preprocess_text(ws):
    """stopwords + lemmatization + alphanumeric + removing ending punctuation"""
    punctuation = ["/",",",".",";",":","(",")","-","[","]"]
    final_word_sentences = []
    for v in tqdm(ws):
        for s in v:
            s = [w for w in s if w not in punctuation]
            s = [w for w in s if not(re.match(r"\d",w))] 
            s = [w for w in s if w not in stopwords.words("portuguese")]
            #s = [nl.lemma(w) for w in s]
            final_word_sentences.append(s)
    final_word_sentences = [[w[:-1] if w[-1] in punctuation else w for w in s] for s in final_word_sentences]
    return final_word_sentences


# Preprocessing cvs - Phrases
def preprocessingCvsPhrases(data, bigram, trigram, out):
    
    data = data.fillna("")
    data.isnull().any()

    start = time()
    punctuation = ["/",",",".",";",":","(",")","-","[","]"]
    data["all"] = data.experiencia.str.cat([data.educacao,data.hab_cmp], sep=' ')
    data["all"] = [word_tokenize(s.lower()) for s in data["all"]]
    data["all"] = [[w for w in s if w not in stopwords.words("portuguese")] for s in data["all"]]
    data["all"] = [[w for w in s if w not in punctuation] for s in data["all"]]
    data["all"] = [[w for w in s if not(re.match(r"\d",w))] for s in data["all"]]
    data["all"] = [[w[:-1] if w[-1] in punctuation else w for w in s] for s in data["all"]]
    print("Elapsed time: %.4f"%(time() - start) )
    data["all"]  = list(trigram[bigram[data["all"]]])
    
    # Saving preprocessed cvs
    f = open(out+"preprocessing/cvs_phrases.series",'wb')
    cvs_phrases = data["all"]
    pickle.dump(cvs_phrases,f)
    #print("CVs saved!")
    
    return data


# Preprocessing jobs - Phrases
def preprocessingJobsPhrases(vagas, bigram, trigram, out):
    
    start = time()
    vagas_ids = vagas["id"].values
    punctuation = ["/",",",".",";",":","(",")","-","[","]"]
    vagas_text = vagas["titulo"].str.cat(vagas["descricao"], sep=" ").values
    vagas_skills = [word_tokenize(v.lower()) for v in vagas_text]
    vagas_skills = [[w for w in s if w not in stopwords.words("portuguese")] for s in vagas_skills]
    vagas_skills = [[w for w in s if w not in punctuation] for s in vagas_skills]
    vagas_skills = [[w for w in s if not(re.match(r"\d",w))] for s in vagas_skills]
    vagas_skills = [[w[:-1] if w[-1] in punctuation else w for w in s] for s in vagas_skills]
    vagas_skills = list(trigram[bigram[vagas_skills]])
    print("Elapsed time: %.4f"%(time() - start) )
    
    # Saving preprocessed job offers 
    f_v = open(out+"preprocessing/vagas_phrases.list",'wb')
    f_iv = open(out+"preprocessing/vagas_phrases_ids.array",'wb')
    pickle.dump(vagas_skills,f_v)
    pickle.dump(vagas_ids,f_iv)
    #print("Vagas saved!")

    
    return vagas_skills, vagas_ids


# Preprocessing cvs - Words
def preprocessingCvsWords(data, out):
    
    # getting one column
    data = data.fillna("")
    data.isnull().any()

    punctuation = ["/",",",".",";",":","(",")","-","[","]"]
    data["all"] = data.experiencia.str.cat([data.educacao,data.hab_cmp], sep=' ')
    data["all"] = [word_tokenize(s.lower()) for s in data["all"]]
    data["all"] = [[w for w in s if w not in stopwords.words("portuguese")] for s in data["all"]]
    data["all"] = [[w for w in s if w not in punctuation] for s in data["all"]]
    #data["all"] = [[nl.lemma(w) for w in s] for s in data["all"]]
    data["all"] = [[w for w in s if not(re.match(r"\d",w))] for s in data["all"]]
    data["all"] = [[w[:-1] if w[-1] in punctuation else w for w in s] for s in data["all"]]

    # Saving preprocessed cvs
    f = open(out+"preprocessing/cvs_words.series",'wb')
    cvs_words = data["all"]
    pickle.dump(cvs_words,f)
    #print("CVs saved!")
    
    return data

# Preprocessing jobs - Words
def preprocessingJobsWords(vagas, out):
    
    # Extracting skills from job offers
    start = time()
    vagas_ids = vagas["id"].values
    punctuation = ["/",",",".",";",":","(",")","-","[","]"]
    vagas_text = vagas["titulo"].str.cat(vagas["descricao"], sep=" ").values
    vagas_skills = [word_tokenize(v.lower()) for v in vagas_text]
    vagas_skills = [[w for w in s if w not in stopwords.words("portuguese")] for s in vagas_skills]
    vagas_skills = [[w for w in s if w not in punctuation] for s in vagas_skills]
    #vagas_skills = [[nl.lemma(w) for w in s] for s in vagas_skills]
    vagas_skills = [[w for w in s if not(re.match(r"\d",w))] for s in vagas_skills]
    vagas_skills = [[w[:-1] if w[-1] in punctuation else w for w in s] for s in vagas_skills]
    print("Elapsed time: %.4f"%(time() - start) )

    # Saving preprocessed vagas 
    f_v = open(out+"preprocessing/vagas_words.list",'wb')
    f_iv = open(out+"preprocessing/vagas_ids.array",'wb')
    pickle.dump(vagas_skills,f_v)
    pickle.dump(vagas_ids,f_iv)
    #print("Vagas saved!")

    return vagas_skills, vagas_ids

    



