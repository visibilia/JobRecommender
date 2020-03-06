#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import re


#cv_id, score
RECOMM_SIZE = 10



def get_format(filenames):
    count = 0 
    df = pd.DataFrame(columns=["cv_id","score"])
    cv_id = 0
    for f in filenames:
        with open(f,errors="ignore") as file:
            for i,l in enumerate(file):
                s = re.findall(r'\d+',l)
                if s:
                    if ((i-1) % RECOMM_SIZE) == 0:
                        cv_id += 1
                    df = df.append({"cv_id":cv_id-1,"score": s[-1]},ignore_index=True)
                    count += 1
    print(count)
    df["score"] = df.score.astype(np.int)
    df["cv_id"] = df.cv_id.astype(np.int)
    return df


def get_results(name, dfi):
    dfo = pd.DataFrame(columns=["method","avg_score","avg_std","avg_precision","avg_me"])
    # MEAN OF SCORE
    dfi["score_freq"] = dfi["score"]/RECOMM_SIZE
    df_avg = dfi.score_freq.mean()
    df_std = dfi.score_freq.std()
    # PRECISION
    val_rec = dfi[dfi.score>=5]
    val_rec_g = val_rec.groupby("cv_id", as_index=0).agg({"score":np.size})
    val_rec_g= val_rec_g.rename(columns={"score":"num_hits"})
    df_avg_precision = val_rec_g.num_hits.mean()/RECOMM_SIZE
    # ME (Minimum Effectiveness)
    n_cvs = dfi.cv_id.nunique()
    val_rec_h = dfi[dfi.score>=5]
    val_rec_h = val_rec_h.groupby("cv_id", as_index=0).agg({"score":np.size})
    val_rec_h= val_rec_h.rename(columns={"score":"num_hits"})
    avg_me = val_rec_h.shape[0]/n_cvs
    
    dfo = dfo.append({"method": name,"avg_score": df_avg, "avg_std": df_std, "avg_precision":df_avg_precision,
                     "avg_me":avg_me },ignore_index=True)
    return dfo

# # TFIDF


root = "tfidf/"
fn = [root + "tf_idf_paul_avaliado.csv", root + "tf_idf_jorge_avaliado.csv", root + "tf_idf_nath_avaliado.csv" ,
     root + "tf_idf_ric_avaliado.csv"]
tfidf = get_format(fn)
tfidf_results = get_results("tfidf",tfidf)
tfidf_results


# # w2vec_cbow


root = "w2vec_cbow/"
fn = [root + "word2vec_cbow_1_jorge_avaliado.csv", root + "word2vec_cbow_4_paul_avaliado.csv", root + "word2vec_cbow_2_nath_avaliado.csv" ,
     root + "word2vec_cbow_3_ric_avaliado.csv"]
w2vec_cbow = get_format(fn)
w2vec_cbow_results = get_results("w2vec_cbow",w2vec_cbow)
w2vec_cbow_results


# # w2vec_skipgram


root = "w2vec_skipgram/"
fn = [root + "w2vec_skip_1_paul_avaliado.csv", root + "w2vec_skip_2_jorge_avaliado.csv", root + "w2vec_skip_3_nath_avaliado.csv" ,
     root + "w2vec_skip_4_ric_avaliado.csv"]
w2vec_skip = get_format(fn)
w2vec_skip_results = get_results("w2vec_skip",w2vec_skip)
w2vec_skip_results


# # w2vec_skipgram_phrases


root = "w2vec_skipgram_phrases/"
fn = [root + "recomm_w2vec_phrases_skg_paul_avaliado.csv", root + "recomm_w2vec_phrases_skg_jorge_avaliado.csv", root + "recomm_w2vec_phrases_skg_nath_avaliado.csv" ,
     root + "recomm_w2vec_phrases_skg_ric_avaliado.csv"]
w2vec_skipf = get_format(fn)
w2vec_skipf_results = get_results("w2vec_skip_phrases",w2vec_skipf)
w2vec_skipf_results


#  # w2vec_cbow_phrases


root = "w2vec_cbow_phrases/"
fn = [root + "recomm_w2vec_phrases_cbow_paul_avaliado.csv", root + "recomm_w2vec_phrases_cbow_jorge_avaliado.csv", root + "recomm_w2vec_phrases_cbow_nath_avaliado.csv" ,
     root + "recomm_w2vec_phrases_cbow_ric_avaliado.csv"]
w2vec_cbowf = get_format(fn)
w2vec_cbowf_results = get_results("w2vec_cbow_phrases",w2vec_cbowf)
w2vec_cbowf_results




