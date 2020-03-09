# Job Recommender System
This repository contains the source code related to the paper intitled "Job Recommendation based on Job Seeker Skills: An Empirical Study" published in the Proceedings of the Text2StoryIR'18 Workshop (Text2Story 2018) , Grenoble, France.

## Framework
* Automatically extract the skills from the job seeker profiles using a variety of text processing techniques. 
* Perform the job recommendation using TF-IDF and four different configurations of Word2vec over a dataset of job seeker profiles and job vacancies collected by us. 
* Experimental evaluation to show the performances of the methods and configurations. 
   - Can be used as a guide to choose the most suitable method and configuration for job recommendation.

## Pipeline

* Data collection

* Data preparation
  * Filtering: filtering out job offers that do not belong to the IT field
    * Use a dictionary of weighted IT terms to match each job offer in its document-like format. 
    * Calculate the weighted sum of the appearances of each word of the job offer in the dictionary and divided it by the appearances of the rest of words in the document (job offer).
    * Get a score with a value from 0 to 1.

  * Text preprocessing: perform stop words removal, tokenization and lemmatization for the Portuguese language.
  * Feature representation: 5 different representations, TF-IDF, Word2Vec using CBOW, Word2Vec
using Skip-Gram, Word2Vec using CBOW with n-grams and Word2Vec using Skip-Gram with n-grams. For the
Word2vec models, a vector space size of 200 was selected after some initial experimentation. Only used the corpus composed by the job offers.

* Recommendation
  * Job matching: given a certain profile, select a group of the nearest job offers based on the distance to that profile
  * Ranking: once retrieved the top "k" job offers for the profile, sort them in descending order based on the inverse of this distance 

* Evaluation

## Code
Argument definitions on Beenet.py

![Arguments](/images/args.png)

Pipeline difinition on Beenet.py

![Pipeline](/images/pipeline.png)
