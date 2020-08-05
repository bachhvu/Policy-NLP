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
             na_values={'date': 'Not specified'}, parse_dates=[7],index_col=0) for f in files)

ann_list = []
pids = []
for f in files:
    df = pd.read_csv(url + f,header=None,names=names,na_values={'date': 'Not specified'},
                 parse_dates=[7],index_col=0)
    ann_list.append(df)
    pids.append(np.int(f.split('_')[0]))
    
annotation = pd.concat(ann_list,axis=0,keys=pids,names=['Policy UID','annotation_id'])
annotation = annotation.drop('policy_id',axis=1)

attr_values = pd.DataFrame(data=None,columns=['annotation_id','start_idx','end_idx','attribute','text','value'])
template = dict.fromkeys('startIndexInSegment endIndexInSegment selectedText value'.split())

import json
import time

#BATCH PROCESSING OF ATTRIBUTE-VALUE PAIRS: THIS NEEDS TO BE DONE MANUALLY!
attr_values1 = pd.DataFrame(data=None,columns=['annotation_id','start_idx','end_idx','attribute','text','value'])
t0 = time.time()
print('Starting at ' + str(t0))
for i in range(0,annotations.shape[0]):    #Note manual adjustment of batch size here...
    
    attr_val = annotations['attributes_value_pairs'].iloc[i]
    ann_id = annotations.index.values[i]
    
    obj = json.loads(attr_val)
    keys = list(obj.keys())
    
    for k in keys:
        obj2 = dict(template, **obj[k]) #Ensures at a minimum we get the empty template data.
        df = pd.DataFrame({'annotation_id': [ann_id],
                           'start_idx': [obj2['startIndexInSegment']],
                           'end_idx': [obj2['endIndexInSegment']],
                           'attribute': [k],
                           'text': [obj2['selectedText']],
                           'value': [obj2['value']]})
        attr_values1 = attr_values1.append(df,ignore_index=True)
t1 = time.time()
print('Finished at ' + str(t1))
print('Total elapsed time ' + str(t1-t0))
attr_values = attr_values.append(attr_values1,ignore_index=True)
attr_values.info()

#Set up indices
attr_values.index.name = 'text_id'
attr_values = attr_values.set_index(['annotation_id'],
                                    append=True).reorder_levels(['annotation_id','text_id'])

attr_values['text'][(attr_values['text'].isnull()) | (attr_values['text'] == 'null')] = 'Not selected'
attr_values['text']

annotations.to_csv("D:/Project/Privacy Policy NLP/Policy-NLP/OPP-115_v1_0/OPP-115/annotations.csv")
attr_values.to_csv("D:/Project/Privacy Policy NLP/Policy-NLP/OPP-115_v1_0/OPP-115/text_selected.csv")

sites = pd.read_csv('D:/Project/Privacy Policy NLP/Policy-NLP/OPP-115_v1_0/OPP-115/documentation/websites_covered_opp115.csv',index_col=3,parse_dates=[4])
sites.head()

sites['In 115 Set?']=sites['In 115 Set?'].apply(lambda yn: True if yn == 'Yes' else False)
sites = sites.loc[sites['In 115 Set?'] == True]
sites.head()

#Reinterpret the categories for the sites. Find the primary category and take the mode across all columns
sectors = sites[sites.columns[6:]]
sectors = sectors.applymap(lambda s: str(s).split(':')[0])
sectors.head()

s = []
for i in range(0,sectors.shape[0]):
    
    sec = sectors.iloc[i][sectors.iloc[i]!='nan'].mode()
    sec = sec.iloc[0] if len(sec) > 0 else 'None'
    s.append(sec)
    
#Drop and append
sites = sites.drop(sites.columns[6:], axis=1)
sites['Sector'] = s

#Grab the policies table
policies = pd.read_csv('D:/Project/Privacy Policy NLP/Policy-NLP/OPP-115_v1_0/OPP-115/documentation/policies_opp115.csv',index_col=0,parse_dates=[2,3])
policies = policies.drop('Unnamed: 4',axis=1)
policies.head()

from os import listdir
import re
from lxml import etree, html
base_dir = 'D:/Project/Privacy Policy NLP/Policy-NLP/OPP-115_v1_0/OPP-115/sanitized_policies/'
files = [f for f in listdir(base_dir) if f.endswith('.html')]

#Check matching policy ids
pids = sites.index.tolist()
for f in files:
    pid = np.int(f.split('_')[0])
    if pid not in pids:
        print('MISSING '+str(pid))

txt = []
pids = []
for f in files:
    with open(base_dir + f, 'r') as pg:
        #print('Reading: ' + f)
        page = pg.read()
        txt.append(page)
        pid = np.int(f.split('_')[0])
        pids.append(pid)
        
#Create dataframe of texts and join it to policies frame
txts = pd.DataFrame({'policy_text': txt},index=pids)
txts.index.name = 'Policy UID'
policies = pd.merge(policies,txts,left_index=True,right_index=True,how='outer')

policies.head()

#Now merge the sites and policies tables
sites = pd.merge(sites,policies,left_index=True,right_index=True,how='outer')
sites.info()

sites.index.nunique()

sites.to_csv("D:/Project/Privacy Policy NLP/Policy-NLP/OPP-115_v1_0/OPP-115/sites.csv")


#First grab policy text from the sites table to parse down
policies = pd.DataFrame(sites['policy_text'])
policies.head()

#For each policy, create a dataframe of segments with unique segment index
dfs = []
for i in range(0,policies.shape[0]):
    tmp = pd.DataFrame(policies['policy_text'].iloc[i].split('|||'),columns=['segments'])
    tmp['Policy UID'] = policies.index.values[i]
    tmp.index.name = 'segment_id'
    dfs.append(tmp)

segments = pd.concat(dfs,axis=0).reset_index().set_index(['Policy UID','segment_id'])

segments.to_csv("D:/Project/Privacy Policy NLP/Policy-NLP/OPP-115_v1_0/OPP-115/segments.csv")
