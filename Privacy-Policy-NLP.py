# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 17:49:59 2020

@author: Bach-PC
"""

#Basic imports
from os import listdir
import pandas as pd
import numpy as np
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.snowball import *
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

url = "D:/Project/Privacy Policy NLP/Policy-NLP/OPP-115_v1_0/OPP-115/annotations/"
files = [f for f in listdir(url) if f.endswith('.csv')]

names = """annotation_id 
        batch_id 
        annotator_id 
        policy_id 
        segment_id 
        category_name 
        attributes_value_pairs 
        date 
        policy_url""".split()

types = {'annotation_id': np.int,
        'batch_id': str,
        'annotator_id': np.int,
        'policy_id': np.int,
        'segment_id': np.int,
        'category_name': str}

annotations = pd.concat(pd.read_csv(url + f,header=None,names=names,
             na_values={'date': 'Not specified'}, parse_dates=[7],index_col=3) for f in files)