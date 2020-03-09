#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from gensim.similarities import WmdSimilarity
from gensim import models as gsm
from time import time
from tqdm import tqdm
import pickle
from DataPreparation.Preprocessing import preprocessingCvsPhrases, preprocessingJobsPhrases

class RecommendationWord2vecPhrases():
    
    def __init__(self, dataPrepFile, dataCvsFile, out):
        self.dataPrepFile = dataPrepFile
        self.dataCvsFile = dataCvsFile
        #n = cvs.shape[0] - 1
        self.n = 50
        self.out = out
        
        
    # Generating recommendations skip-gram
    def recommendation(self,cvs,vagas_ti,vagas_ids,num_best, instance, model):
        
        df_final = pd.DataFrame(columns=["exp","edu","hab","v_tit","v_desc","sim"])
        for j in tqdm(range(self.n)):
            df_aux = pd.DataFrame()
            #index = np.random.randint(n)
            query = cvs["all"][j]
            sims = instance[query] 
            aux_sims = []
            aux_vids = []
            for i in range(num_best):
                aux_sims.append(round(sims[i][1],4))
                aux_vids.append(vagas_ids[sims[i][0]]) 
            aux_exp = ["the same"] * num_best
            aux_edu = ["the same"] * num_best
            aux_hab = ["the same"] * num_best
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
        df_final.to_csv(self.out+"recommendations/recomendacoes_finais_phrases_"+model+".csv")
        print("Recommendation "+model+" done!")
        
        
    def main(self):    
        
        print("Recommendation using Embeddings-Phrases")
        
        # Loading preprocessed data
        vagas_ti = pd.read_csv(self.dataPrepFile)
        
        # Loading cvs data
        cvs = pd.read_csv(self.dataCvsFile)
        
        # Loading bigram and trigrams
        bigram = pickle.load(open(self.out+"wordEmbeddings/vagas_cv.bigram","rb"))
        trigram = pickle.load(open(self.out+"wordEmbeddings/vagas_cv.trigram","rb"))
        
        # Preprocessing cvs
        cvs = preprocessingCvsPhrases(cvs, bigram, trigram, self.out)
        
        # Preprocessing job offers
        vagas_skills, vagas_ids = preprocessingJobsPhrases(vagas_ti, bigram, trigram, self.out)
        
        # Loading model
        model_skill_skg = gsm.Word2Vec.load(self.out+"wordEmbeddings/ti_skill_phrases_skg.model")
        model_skill_cbow = gsm.Word2Vec.load(self.out+"wordEmbeddings/ti_skill_phrases_cbow.model")
        
        # Using similarity framework for Word Mover's Distance (WMD)
        num_best = 10
        start = time()
        #Normalizing word2vec vectors
        model_skill_skg.init_sims(replace=True)
        instance_skg = WmdSimilarity(vagas_skills, model_skill_skg, num_best=num_best)
        
        model_skill_cbow.init_sims(replace=True)
        instance_cbow = WmdSimilarity(vagas_skills, model_skill_cbow, num_best=num_best)
        print("Time: %.4f" %(time()-start))
        
        self.recommendation(cvs, vagas_ti, vagas_ids, num_best, instance_skg, "skg")
        self.recommendation(cvs, vagas_ti, vagas_ids, num_best, instance_cbow, "cbow")
        
        print("Recommendation using Embeddings-Phrases done!")
    
