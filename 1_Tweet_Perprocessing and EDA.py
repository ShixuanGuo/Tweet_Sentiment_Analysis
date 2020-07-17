#!/usr/bin/env python
# coding: utf-8

# # Part 1 Preprocessing

# ## Packages

# In[ ]:


# analysis packages
import sys

get_ipython().system('{sys.executable} -m pip install numpy')
import numpy as np 

get_ipython().system('{sys.executable} -m pip install pandas')
import pandas as pd

get_ipython().system('{sys.executable} -m pip install nltk')
import nltk

import warnings
warnings.simplefilter(action='ignore')

get_ipython().system('{sys.executable} -m pip install -U textblob')


get_ipython().run_line_magic('run', './Text_Normalization_Function.ipynb')


#Plot packages
import matplotlib.pyplot as plt
get_ipython().run_line_magic('pip', 'install plotly==4.7.1')
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
from plotly import tools
from plotly.subplots import make_subplots

get_ipython().run_line_magic('pip', 'install cufflinks')
import cufflinks
cufflinks.go_offline()
cufflinks.set_config_file(world_readable=True, theme='pearl')

#Others
import re
import string


# ## Read data

# In[2]:


# Import data
df_train=pd.read_csv('tweet-sentiment-extraction/train.csv')
df_test=pd.read_csv('tweet-sentiment-extraction/test.csv')
df_submission=pd.read_csv('tweet-sentiment-extraction/sample_submission.csv')


# # Part 2 EDA

# ### A quick look at the data and sentiment

# In[5]:


# describe training data
df_train.describe()


# In[ ]:


# count number of records by sentiment
df_train.groupby('sentiment').count()


# ## 2.1 Missing Data

# In[44]:


null_train_data=df_train[df_train.isnull().any(axis=1)]
null_train_data


# In[3]:


df_train=df_train.fillna('')
df_test=df_test.fillna('')


# ## 2.2 Length

# #### Calculate the length of text and selected text 

# In[6]:


# count number of words for each text of training data
num_words_text = []
for num in df_train['text']:
    words_text = len(num.split())
    num_words_text.append(words_text)

df_train["num_words_text"] = num_words_text


# In[7]:


# count number of words for each selected_text of training data
num_words_s_text = []
for num2 in df_train['selected_text']:
    words_s_text = len(num2.split())
    num_words_s_text.append(words_s_text)

df_train["num_words_s_text"] = num_words_s_text


# In[8]:


# calculate difference of # between text and selected_text of training data
df_train["difference"] = df_train["num_words_text"]-df_train["num_words_s_text"]
# check average length and difference
df_train.groupby('sentiment').mean()
# plot histogram of difference
plt.hist(df_train["difference"])
plt.show()


# ## 2.3 Punctuation

# ### Find punctuation

# In[ ]:


def find_punctuation(string):
    punctuations=re.findall(r'[!"\$%&\'()*+,\-.\/:;=#@?\[\\\]^_`{|}~]*', string)
    text="".join(punctuations)
    return list(text)


# In[19]:


#find punctuations and the length of punctuations in text and selected text
df_train['target_punctuation']=df_train['Target'].apply(lambda x: find_punctuation(x))
df_train['target_punct_len']=df_train['Target'].apply(lambda x:len(find_punctuation(x)))

df_train['text_punctuation']=df_train['text'].apply(lambda x: find_punctuation(x))
df_train['text_punct_len']=df_train['text'].apply(lambda x:len(find_punctuation(x)))


# ### Descriptibe Analysis of Selected_text's Punctuation

# #### 1. Overview

# In[32]:


df_train['target_punctuation'].head()
df_train['target_punctuation'].value_counts()


# #### 2. Frequency of unique punctuation

# In[74]:


#Get the frequency of unique punctuation-using all sentiment text
target_punct=pd.Series([item for sublist in df_train.target_punctuation for item in sublist])
target_punct_df=target_punct.groupby(target_punct).size().rename_axis('punct').reset_index(name='num').sort_values(by=['num'], ascending=False).reset_index(drop=True)
target_punct_df['frequency']=round(target_punct_df['num']/len(tweet_train)*100,2)
target_punct_df


# ### Difference between selected-text's punctuation and original text's punctuation

# #### 1. Overview

# In[20]:


def find_difference(text, target):
    list=[]
    for i in text:
        if i not in target:
            list.append(i)
    return list


# In[21]:


df_train['punct_difference']=df_train.apply(lambda x: find_difference(x['text_punctuation'],x['target_punctuation']), axis=1)
df_train['punct_difference'].value_counts()


# #### 2. Frequency of unique punctuation 

# In[72]:


#Get the frequency of unique punctuation which are not included in selected text-using all sentiment text
diff_punct=pd.Series([item for sublist in df_train.punct_difference for item in sublist])
diff_punct_df=diff_punct.groupby(target_punct).size().rename_axis('diff_punct').reset_index(name='num').sort_values(by=['num'], ascending=False).reset_index(drop=True)
diff_punct_df['frequency']=round(diff_punct_df['num']/len(tweet_train)*100,2)
diff_punct_df


# ### Punctuation analysis in terms of sentiment

# #### 1. How many selected texts do not have any punctuation in terms of sentiment

# In[92]:


df_punct=pd.DataFrame(df_train.loc[df_train['target_punctuation'].str.len() == 0]['sentiment'].value_counts()).reset_index()
df_punct.rename(columns={"index": "sentiment", "sentiment": "no_punct_count"})


# #### 2. How many selected texts do not have different punctuations with text in terms of sentiment

# In[91]:


df_punct1=pd.DataFrame(df_train.loc[df_train['punct_difference'].str.len() == 0]['sentiment'].value_counts()).reset_index()
df_punct1.rename(columns={"index": "sentiment", "sentiment": "no_punct_count"})


# #### 3. Populary punctuations analysis

# In[94]:


df_punct2_dist=pd.DataFrame(df_train,columns=['target_punctuation','sentiment'])
df_punct2_dist=df_punct2_dist[df_punct2_dist['target_punctuation'].map(lambda d: len(d)) > 0]
df_punct2_dist=df_punct2_dist.explode('target_punctuation')
df_punct2_dist.head()


# In[109]:


df_punct2_positive=pd.DataFrame(df_punct2_dist.loc[df_punct2_dist['sentiment']=="positive"]['target_punctuation'].value_counts()).reset_index().rename(columns={'index': 'punct','target_punctuation':'pos_punct'})
df_punct2_negative=pd.DataFrame(df_punct2_dist.loc[df_punct2_dist['sentiment']=="negative"]['target_punctuation'].value_counts()).reset_index().rename(columns={'index': 'punct','target_punctuation':'neg_punct'})
df_punct2_neutral=pd.DataFrame(df_punct2_dist.loc[df_punct2_dist['sentiment']=="neutral"]['target_punctuation'].value_counts()).reset_index().rename(columns={'index': 'punct','target_punctuation':'neut_punct'})
df_punct2_positive.head()


# #### plot punctuation distributions in terms of sentiment

# In[111]:


fig = make_subplots(rows=1, cols=3)

fig.append_trace(go.Bar(x=df_punct2_positive.punct[:10],y=df_punct2_positive.pos_punct[:10],name='Positive',marker_color='green'), row=1, col=1)
fig.append_trace(go.Bar(x=df_punct2_negative.punct[:10],y=df_punct2_negative.neg_punct[:10],name='Negative',marker_color='red'), row=1, col=2)
fig.append_trace(go.Bar(x=df_punct2_neutral.punct[:10],y=df_punct2_neutral.neut_punct[:10],name='Neutral',marker_color='orange'), row=1, col=3)

fig.update_xaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_yaxes(showline=True, linewidth=2, linecolor='black', mirror=True)
fig.update_layout(title_text="Selected Text - Sentiment vs Punctuation",title_x=0.5)
fig.show()


# #### How many selected texts have popular punctuation . in terms of sentiment

# In[136]:


def find_popular_punct(string):
    punctuations=re.findall(r'[.]{3,6}', string)
    text="".join(punctuations)
    return list(text)


# In[131]:


tweet_train['star']=tweet_train['Target'].apply(lambda x:find_popular_punct(x))
tweet_train.loc[tweet_train['star']!=0]['sentiment'].value_counts().to_frame()


# ## 2.4 Part of Speech

# #### Add text tokens, text tags, selected_text tokens and selected_text tags to the dataframe

# In[13]:


# add tokens
df_train["text_tokens"] = ""
df_train["text_tags"] = ""
df_train["s_text_tokens"] = ""
df_train["s_text_tags"] = ""

for index,row in df_train.iterrows():
    df_train["text_tokens"][index] = nltk.word_tokenize(row[1])
    df_train["s_text_tokens"][index] = nltk.word_tokenize(row[2])


# In[14]:


# add tags of parts of speech
for index,row in df_train.iterrows():
    df_train["text_tags"][index] = []
    df_train["s_text_tags"][index] = []
    for i in nltk.pos_tag(row[7]):
        df_train["text_tags"][index].append(i[1])
    for j in nltk.pos_tag(row[9]):
        df_train["s_text_tags"][index].append(j[1])


# In[15]:


# define list of parts of speech
Tag_List = ["CC","IN","JJ","JJR","JJS","NN","NNS","NNP","NNPS","PRP","PRP$","WP","WP$","RB","RBR","RBS","VB",
            "VBD","VBG","VBN","VBP","VBZ","UH","CD","DT","EX","FW","LS","MD","PDT","POS","RP","TO","WDT","WRB"]

# define dictionary (class) of parts of speech
Tag_Dict = {"conjunction":["CC","IN"], "adjective":["JJ","JJR","JJS"], "noun":["NN","NNS","NNP","NNPS"], 
            "pronoun":["PRP","PRP$","WP","WP$"], "adverb":["RB","RBR","RBS"], 
            "verb":["VB","VBD","VBG","VBN","VBP","VBZ"], "interjection":["UH"], 
            "others":["CD","DT","EX","FW","LS","MD","PDT","POS","RP","TO","WDT","WRB"]}


# In[16]:


# remove noises of tags
for index,row in df_train.iterrows():
    for i in row[8]:
        if i not in Tag_List:
            df_train["text_tags"][index].remove(i)
    for j in row[10]:
        if j not in Tag_List:
            df_train["s_text_tags"][index].remove(j)


# #### Classify parts of speech with easy-to-understand expressions

# In[18]:


# define a function to classify each part of speech
def identify_class(str_list):
    class_list = []
    for i in str_list:
        if i in ["CC","IN"]:
            class_list.append("conjunction")
        elif i in ["JJ","JJR","JJS"]:
            class_list.append("adjective")
        elif i in ["NN","NNS","NNP","NNPS"]:
            class_list.append("noun")
        elif i in ["PRP","PRP$","WP","WP$"]:
            class_list.append("pronoun")
        elif i in ["RB","RBR","RBS"]:
            class_list.append("adverb")
        elif i in ["VB","VBD","VBG","VBN","VBP","VBZ"]:
            class_list.append("verb")
        elif i == "UH":
            class_list.append("interjection")
        else:
            class_list.append("others")
    return class_list


# In[19]:


# apply the function to the dataset
df_train["text_class"] = ""
df_train["s_text_class"] = ""
df_train["text_class"] = df_train["text_tags"].apply(identify_class)
df_train["s_text_class"] = df_train["s_text_tags"].apply(identify_class)
df_train.head(2)


# #### Analyze the distribution of parts of speech for the dataset

# In[20]:


# consolidate text data
text_all_class = []
s_text_all_class = []

for i in df_train["text_class"]:
    text_all_class.extend(i)

for i in df_train["s_text_class"]:
    s_text_all_class.extend(i)


# In[21]:


# view number of words for text and selected text
print("Number of records of text:", len(text_all_class),
      "\nNumber of records of selected_text:", len(s_text_all_class))


# In[22]:


# view proportions of the selected_text in text by parts of speech
part_of_speech = ["adjective","adverb","conjunction","interjection","noun","others","pronoun","verb"]
group_text_all_class = pd.value_counts(text_all_class).sort_index(ascending=True)
group_s_text_all_class = pd.value_counts(s_text_all_class).sort_index(ascending=True)
print(group_s_text_all_class/group_text_all_class)


# In[23]:


# visualize number of records of selected_text and text by parts of speech
barWidth = 0.25
fig = plt.figure(figsize=(12,4))

r1 = np.arange(len(group_text_all_class))
r2 = [x + barWidth for x in r1]
 
plt.bar(r1, group_text_all_class, color='#9494b8', width=barWidth, edgecolor='white', label='text')
plt.bar(r2, group_s_text_all_class, color='#c2c2d6', width=barWidth, edgecolor='white', label='s_text')

plt.title("All Tweet", fontweight='bold')
plt.xlabel('part of speech')
plt.xticks([r + barWidth for r in range(len(group_text_all_class))], part_of_speech)
 
plt.legend()
plt.show()


# #### Analyze the distribution of parts of speech by sentiment

# In[24]:


# list of text and selected_text of pos/neu/neg data
pos_text_all_class = []
pos_s_text_all_class = []

neu_text_all_class = []
neu_s_text_all_class = []

neg_text_all_class = []
neg_s_text_all_class = []

# consolidate text data
for i in df_train[df_train["sentiment"] == "positive"]["text_class"]:
    pos_text_all_class.extend(i)
for i in df_train[df_train["sentiment"] == "positive"]["s_text_class"]:
    pos_s_text_all_class.extend(i)
    
for i in df_train[df_train["sentiment"] == "neutral"]["text_class"]:
    neu_text_all_class.extend(i)
for i in df_train[df_train["sentiment"] == "neutral"]["s_text_class"]:
    neu_s_text_all_class.extend(i)
    
for i in df_train[df_train["sentiment"] == "negative"]["text_class"]:
    neg_text_all_class.extend(i)
for i in df_train[df_train["sentiment"] == "negative"]["s_text_class"]:
    neg_s_text_all_class.extend(i)


# In[25]:


# view proportions of the selected_text in text [POSITIVE]
pos_group_text_all_class = pd.value_counts(pos_text_all_class).sort_index(ascending=True)
pos_group_s_text_all_class = pd.value_counts(pos_s_text_all_class).sort_index(ascending=True)
print(pos_group_s_text_all_class/pos_group_text_all_class)


# In[26]:


# visualize number of records of selected_text and text [POSITIVE]
barWidth = 0.25
fig = plt.figure(figsize=(12,4))

r1 = np.arange(len(pos_group_text_all_class))
r2 = [x + barWidth for x in r1]
 
plt.bar(r1, pos_group_text_all_class, color='#79d279', width=barWidth, edgecolor='white', label='text')
plt.bar(r2, pos_group_s_text_all_class, color='#c6ecc6', width=barWidth, edgecolor='white', label='s_text')

plt.title("Positive Tweet", fontweight='bold')
plt.xlabel('part of speech')
plt.xticks([r + barWidth for r in range(len(pos_group_text_all_class))], part_of_speech)
 
plt.legend()
plt.show()


# In[27]:


# view proportions of the selected_text in text [NEUTRAL]
neu_group_text_all_class = pd.value_counts(neu_text_all_class).sort_index(ascending=True)
neu_group_s_text_all_class = pd.value_counts(neu_s_text_all_class).sort_index(ascending=True)
print(neu_group_s_text_all_class/neu_group_text_all_class)


# In[28]:


# visualize number of records of selected_text and text [NEUTRAL]
barWidth = 0.25
fig = plt.figure(figsize=(12,4))

r1 = np.arange(len(neu_group_text_all_class))
r2 = [x + barWidth for x in r1]
 
plt.bar(r1, neu_group_text_all_class, color='#d2a679', width=barWidth, edgecolor='white', label='text')
plt.bar(r2, neu_group_s_text_all_class, color='#e6ccb3', width=barWidth, edgecolor='white', label='s_text')

plt.title("Neutral Tweet", fontweight='bold')
plt.xlabel('part of speech')
plt.xticks([r + barWidth for r in range(len(neu_group_text_all_class))], part_of_speech)
 
plt.legend()
plt.show()


# In[29]:


# view proportions of the selected_text in text [NEGATIVE]
neg_group_text_all_class = pd.value_counts(neg_text_all_class).sort_index(ascending=True)
neg_group_s_text_all_class = pd.value_counts(neg_s_text_all_class).sort_index(ascending=True)
print(neg_group_s_text_all_class/neg_group_text_all_class)


# In[30]:


# visualize number of records of selected_text and text [NEGATIVE]
barWidth = 0.25
fig = plt.figure(figsize=(12,4))

r1 = np.arange(len(neg_group_text_all_class))
r2 = [x + barWidth for x in r1]
 
plt.bar(r1, neg_group_text_all_class, color='#ff4d4d', width=barWidth, edgecolor='white', label='text')
plt.bar(r2, neg_group_s_text_all_class, color='#ffb3b3', width=barWidth, edgecolor='white', label='s_text')

plt.title("Negative Tweet", fontweight='bold')
plt.xlabel('part of speech')
plt.xticks([r + barWidth for r in range(len(neg_group_text_all_class))], part_of_speech)
 
plt.legend()
plt.show()

