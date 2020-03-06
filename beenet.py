#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 09:51:40 2020

@author: fabiana
"""
from DataPreparation.Filtering import Filtering
from WordEmbeddings.WordEmbeddingPhrases import WordEmbPhrases
from WordEmbeddings.WordEmbeddingWords import WordEmbWords
from Recommendation.RecommendationW2vecPhrases import RecommendationWord2vecPhrases
from Recommendation.RecommendationW2vecWords import RecommendationWord2vecWords
from Recommendation.RecommendationTF_IDF import RecommendationTF_IDF
import argparse
import os 


def main():

    parser = argparse.ArgumentParser(description = 'Beenet Code.')

    parser.add_argument('-o', action = 'store', dest = 'out', default = os.getcwd()+"/out/", required = False,
                    help = 'Output directory.')
    parser.add_argument('-i', action = 'store', dest = 'input', default = os.getcwd()+"/Data/", required = False,
                    help = 'Input directory.')
    parser.add_argument('-dc', action = 'store', dest = 'datacatho', default = "Data/vagas_catho_etiquetadas.csv", required = False,
                    help = 'Directory of Catho data.')
    parser.add_argument('-dl', action = 'store', dest = 'datalinkedin', default = "Data/cvs_linkedin.csv", required = False,
                    help = 'Directory of Linkedin data.')

    arguments = parser.parse_args()

    return arguments
 
  
if __name__== "__main__":
    
  arguments = main()
  
  # Calling filtering step
  f = Filtering(arguments.datacatho, arguments.out)
  f.applyLimiar()
  
  # Calling feature representation step (including preprocessing) - Word Embeddings-Phrases
  wbp = WordEmbPhrases(arguments.out+"/vagas_ti.csv", arguments.out)
  wbp.main()
  
  # Calling feature representation step (including preprocessing) - Word Embeddings-Words
  wbw = WordEmbWords(arguments.out+"/vagas_ti.csv", arguments.out)
  wbw.getWordEmbModel()
  
  # Recommendation using Embeddings-Phrases
  recPhrases = RecommendationWord2vecPhrases(arguments.out+"/vagas_ti.csv", arguments.datalinkedin, "out/")
  recPhrases.main()
  
  # Recommendation using Embeddings-Words
  recWords = RecommendationWord2vecWords(arguments.out+"/vagas_ti.csv", arguments.datalinkedin, "out/")
  recWords.main()
  
  # Recommendation using TF_IDF
  recTf_idf = RecommendationTF_IDF(arguments.out+"/vagas_ti.csv", arguments.datalinkedin, "out/")
  recTf_idf.main()

  # Generating Final Results  
  
  
  
  
  
  