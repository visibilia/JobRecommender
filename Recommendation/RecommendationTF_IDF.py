#!/usr/bin/env python
# coding: utf-8


import pandas as pd
from gensim.similarities import MatrixSimilarity
from gensim import models as gsm
from gensim import corpora as gcorp
from tqdm import tqdm
import pickle

class RecommendationTF_IDF():
    
    def __init__(self, dataPrepFile, dataCvsFile, out):
        self.dataPrepFile = dataPrepFile
        self.dataCvsFile = dataCvsFile
        #n = cvs.shape[0] - 1
        self.n = 50
        self.out = out
        
    def recommendationTf_idf(self,cvs,vagas_ti,vagas_ids,cvs_words,dictionary,tfidf,index):
 
        df_final = pd.DataFrame(columns=["exp","edu","hab","v_tit","v_desc","sim"])
        for j in tqdm(range(self.n)):
            df_aux = pd.DataFrame()
            query = cvs_words[j]
            query = dictionary.doc2bow(query)
            query = tfidf[query]
            sims = index[query] 
            aux_sims = []
            aux_vids = []
            for i in range(10):
                aux_sims.append(round(sims[i][1],4))
                aux_vids.append(vagas_ids[sims[i][0]]) 
            aux_exp = ["the same"] * 10
            aux_edu = ["the same"] * 10
            aux_hab = ["the same"] * 10
            aux_exp[0] = cvs.iloc[j]["experiencia"]
            aux_edu[0] = cvs.iloc[j]["educacao"]
            aux_hab[0] = cvs.iloc[j]["hab_cmp"]
            df_aux["exp"] = pd.Series(aux_exp) 
            df_aux["edu"] = pd.Series(aux_edu) 
            df_aux["hab"] = pd.Series(aux_hab) 
            df_aux["v_tit"] = vagas_ti[vagas_ti.id.isin(aux_vids)]["titulo"].values    
            df_aux["v_desc"] = vagas_ti[vagas_ti.id.isin(aux_vids)]["descricao"].values
            df_aux["sim"] = pd.Series(aux_sims)
            df_final = pd.concat([df_final,df_aux], axis=0)
        df_final.to_csv(self.out+"recommendations/recomendacoes_finais_tfidf.csv")
        #print("Done!")


    def main(self):
        
        print("Recommendation using TF_IDF")
        
        # Loading preprocessed data
        vagas_ti = pd.read_csv(self.dataPrepFile)
        vagas_ids = pickle.load(open(self.out+"preprocessing/vagas_ids.array","rb"))
        vagas_words = pickle.load(open(self.out+"preprocessing/vagas_words.list","rb"))
        cvs_words = pickle.load(open(self.out+"preprocessing/cvs_words.series","rb"))
        cvs = pd.read_csv(self.dataCvsFile)
        cvs = cvs.fillna("")
        cvs.isnull().any()
        #print("Loading cvs done!")

        # Creating a dictionary 
        dictionary = gcorp.Dictionary(vagas_words)
        dictionary.save(self.out+'preprocessing/tf_idf/vagas.dict') # store the dictionary, for future reference
        
        # compile corpus (vectors number of times each elements appears)
        raw_corpus = [dictionary.doc2bow(v) for v in vagas_words]
        gcorp.MmCorpus.serialize(self.out+'preprocessing/tf_idf/vagas.mm', raw_corpus) # store to disk
        print("Tamanho do dicionário: "+str(len(dictionary)))
        
        # STEP 2 : similarity between corpuses
        dictionary = gcorp.Dictionary.load(self.out+'preprocessing/tf_idf/vagas.dict')
        corpus = gcorp.MmCorpus(self.out+'preprocessing/tf_idf/vagas.mm')

        # Transform Text with TF-IDF
        tfidf = gsm.TfidfModel(corpus) # step 1 -- initialize a model

        # corpus tf-idf
        corpus_tfidf = tfidf[corpus] 

        # STEP 3 : Create similarity matrix of all files
        index = MatrixSimilarity(corpus_tfidf,num_features=len(dictionary),num_best=10)
        index.save(self.out+'preprocessing/tf_idf/vagas.index')
        index = MatrixSimilarity.load(self.out+'preprocessing/tf_idf/vagas.index')

        self.recommendationTf_idf(cvs,vagas_ti,vagas_ids,cvs_words,dictionary,tfidf,index)
        
        print("Recommendation using TF_IDF done!")
         
        
        