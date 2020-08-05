# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# <a href="https://colab.research.google.com/github/bachvu98/Policy-NLP/blob/master/Preprocessing_Policy.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
# %% [markdown]
# First, we import the require dependencies

# %%
import pandas as pd
import numpy as np
import string
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')
nltk.download('punkt')

# %% [markdown]
# Read in the data from GitHub repository

# %%
annotations = pd.read_csv('https://raw.githubusercontent.com/bachvu98/Policy-NLP/master/OPP-115_v1_0/OPP-115/annotations.csv')
sites = pd.read_csv('https://raw.githubusercontent.com/bachvu98/Policy-NLP/master/OPP-115_v1_0/OPP-115/sites.csv')
segments = pd.read_csv('https://raw.githubusercontent.com/bachvu98/Policy-NLP/master/OPP-115_v1_0/OPP-115/segments.csv')

# %% [markdown]
# Preview of **annotations** and **segments** table

# %%
annotations.head()


# %%
segments.head()

# %% [markdown]
# Merge annotations to corresponding segments

# %%
joined = pd.merge(annotations,segments,on=['Policy UID','segment_id'],how='outer')
joined['category_name'] = joined['category_name'].fillna(value='None')
joined = joined.drop(['batch_id','attributes_value_pairs','date','annotation_id','annotator_id','policy_url'],axis=1)
#joined = seg_ind.merge(ann_ind)
print(joined.shape)
joined.head()

# %% [markdown]
# There are usually cases where a single segment belong to multiple categories.

# %%
print(joined.groupby(['Policy UID','segment_id']).agg(lambda x: x.nunique())['category_name'])

# %% [markdown]
# In this case, we select the category name that appears most often in each segment.

# %%
#Get the mode of each segment
mode_categories = joined.groupby(['Policy UID','segment_id']).agg(lambda x: x.value_counts().index[0])
mode_categories = mode_categories.reset_index()
mode_categories.head()

# %% [markdown]
# # Preprocessing segments text

# %%
def clean_text(text):
  text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", text).split())
  #Then tokenisation
  tokens = word_tokenize(text)
  # convert to lower case
  tokens = [w.lower() for w in tokens]
  # remove punctuation from each word
  table = str.maketrans('', '', string.punctuation)
  stripped = [w.translate(table) for w in tokens]
  # remove remaining tokens that are not alphabetic
  words = [word for word in stripped if word.isalpha()]
  # filter out stop words
  stop_words = set(stopwords.words('english'))
  # You can add more stop words here, specific for tweets
  words = [w for w in words if not w in stop_words]
  # stemming of words
  porter = PorterStemmer()
  words = [porter.stem(word) for word in words]
  # Convert from list to a sentence again
  text = ' '.join(word for word in words)
  return text


# %%
#Process the segments here
mode_categories['segments'] = mode_categories['segments'].apply(clean_text)


# %%
mode_categories.head()


# %%
categories = list(mode_categories['category_name'].unique())
print(categories)
cols = {'Other': 'other',
        'Policy Change': 'policy_change',
        'First Party Collection/Use': 'first_party_collection_use',
        'Third Party Sharing/Collection': 'third_party_sharing_collection',
        'Do Not Track': 'do_not_track',
        'User Choice/Control': 'user_choice_control',
        'International and Specific Audiences': 'international_specific_audiences',
        'Data Security': 'data_security',
        'Data Retention': 'data_retention',
        'User Access, Edit and Deletion': 'user_access_edit_deletion'}


# %%
#Loop through the categories and generate a set of new columns with names in cols
binary_categories = pd.DataFrame({'Policy UID':mode_categories['Policy UID'], 'segment_id':mode_categories['segment_id'], 'segments':mode_categories['segments']})


# %%
for category in categories:
    one_hot = lambda s: 1 if s.startswith(category) else 0
    binary_categories[cols[category]] = mode_categories['category_name'].apply(one_hot)


# %%
binary_categories.head()


# %%
binary_categories.to_csv("/content/drive/My Drive/OPP-115/OPP-115/binary_segment_categories.csv")


