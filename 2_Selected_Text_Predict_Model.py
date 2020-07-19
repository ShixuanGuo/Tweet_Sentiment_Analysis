#!/usr/bin/env python
# coding: utf-8

# # Part 3 Model Construction

# In[ ]:


#library

# sklearn 
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

# nltk
from nltk.corpus import stopwords
from nltk.util import ngrams
stop=set(stopwords.words('english'))

get_ipython().run_line_magic('pip', 'install wordcloud')
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image

# Avoid warning
import warnings
warnings.filterwarnings("ignore")


# ## 3.1 Preprocessing

# ### 1. Convert sentiment

# In[ ]:


# convert sentiment
def convert_sentiment(text):
    if text=='positive':
        return 1
    elif text=='negative':
        return -1
    else:
        return 0


# ### 2. Split data by sentiment

# In[4]:


# df subsets by sentiment
df_positive = df_train[df_train["sentiment"]=="positive"]
df_positive = df_positive.reset_index(drop=True)

df_neutral = df_train[df_train["sentiment"]=="neutral"]
df_neutral = df_neutral.reset_index(drop=True)

df_negative = df_train[df_train["sentiment"]=="negative"]
df_negative = df_negative.reset_index(drop=True)


# ### 3. Split train and test dataset

# In[9]:


# train-test split
train_train=df_train.sample(frac=0.7, random_state=7)
test_train=df_train.drop(train_train.index)

train_train=train_train.reset_index(drop=True)
test_train=test_train.reset_index(drop=True)


# #### In terms of sentiment

# In[ ]:


# train and test data with a random seed
train_pos = df_positive.sample(frac=0.7, random_state=7)
test_pos = df_positive.drop(train_pos.index)

train_neu = df_neutral.sample(frac=0.7, random_state=7)
test_neu = df_neutral.drop(train_neu.index)

train_neg = df_negative.sample(frac=0.7, random_state=7)
test_neg = df_negative.drop(train_neg.index)


# In[ ]:


train_pos=train_pos.reset_index(drop=True)
test_pos=test_pos.reset_index(drop=True)

train_neu=train_neu.reset_index(drop=True)
test_neu=test_neu.reset_index(drop=True)

train_neg=train_neg.reset_index(drop=True)
test_neg=test_neg.reset_index(drop=True)


# ## 3.2 Select input

# ### 1. Get subset

# In[6]:


def get_subset_list(text):
    corpus = text.strip().split(" ")
    subset= []
    k=len(corpus)
    for i in range(1,8):
        if i<=k:
            for j in range(k-i+1):
                temp=[]
                temp.extend(corpus[n] for n in range(j,j+i))
                subset.append(' '.join(temp))
    return subset


# ### 2. TF-IDF vectorization

# In[8]:


from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=1000) 


# In[33]:


# train set
train_text_matrix = vectorizer.fit_transform(train_train['text'].values.astype('U')).toarray()
train_matrix_names = vectorizer.get_feature_names() 

train_text_matrix_table = pd.DataFrame(np.round(train_text_matrix, 2), columns = train_matrix_names)
train_text_matrix_table.head()


# In[34]:


#test set
test_text_matrix = vectorizer.transform(test_train['text'].values.astype('U')).toarray()
test_matrix_names = vectorizer.get_feature_names() 

test_text_matrix_table = pd.DataFrame(np.round(test_text_matrix, 2), columns = test_matrix_names)
test_text_matrix_table.head()


# ### 3. Start-end-position (Encode)

# In[6]:


import pandas as pd, numpy as np

get_ipython().run_line_magic('pip', 'install --upgrade tensorflow')
import tensorflow as tf

get_ipython().system('{sys.executable} -m pip install keras')
import tensorflow.keras.backend as K
from sklearn.model_selection import StratifiedKFold

get_ipython().run_line_magic('pip', 'install transformers')
from transformers import *
from transformers import BertTokenizer,BertConfig,TFBertModel

get_ipython().system('{sys.executable} -m pip install tokenizers')
import tokenizers

print('TF version',tf.__version__)


# In[8]:


MAX_LEN = 96
tokenizer = tokenizers.ByteLevelBPETokenizer(
    vocab_file='vocab-roberta-base.json', 
    merges_file='merges-roberta-base.txt', 
    lowercase=True,
    add_prefix_space=True
)


# In[11]:


sentiment_id = {'positive': 1313, 'negative': 2430, 'neutral': 7974}


# #### Total train data

# Train input and output

# In[12]:


ct = train_train.shape[0]
input_ids = np.ones((ct,MAX_LEN),dtype='int32')
attention_mask = np.zeros((ct,MAX_LEN),dtype='int32')
#train output
token_type_ids = np.zeros((ct,MAX_LEN),dtype='int32')
start_tokens = np.zeros((ct,MAX_LEN),dtype='int32')
end_tokens = np.zeros((ct,MAX_LEN),dtype='int32')


# In[24]:


#choose output: token_type, one_output, two_output (start_position, start_token)
def get_encoded_train(train_train):
    ct = train_train.shape[0]
    input_ids = np.ones((ct,MAX_LEN),dtype='int32')
    attention_mask = np.zeros((ct,MAX_LEN),dtype='int32')
    outputs=np.zeros((ct,2),dtype='int32')
    
    #start_tokens = np.zeros((ct,MAX_LEN),dtype='int32')
    #end_tokens = np.zeros((ct,MAX_LEN),dtype='int32')

    #start_position=[]
    #end_position=[]
    #token_type_ids = np.zeros((ct,MAX_LEN),dtype='int32')
    
    for k in range(train_train.shape[0]):

    #1. FIND OVERLAP
    #input: text and selected text output: chars: 0/1 (1: overlap position in terms of character)
        text1 = " "+" ".join(train_train.loc[k,'text'].split())
        text2 = " ".join(train_train.loc[k,'selected_text'].split())
        idx = text1.find(text2) #first position of overlap
        chars = np.zeros((len(text1)))
        chars[idx:idx+len(text2)]=1 #output
        if text1[idx-1]==' ': chars[idx-1] = 1 
        enc = tokenizer.encode(text1) 

    #2. ID_OFFSETS
    #ids is the numeric id of encoded text； get the start and end position of each word in terms of character
        offsets = []; idx=0
        for t in enc.ids:
            w = tokenizer.decode([t])
            offsets.append((idx,idx+len(w)))
            idx += len(w)

    #3. Generate inputs and outputs
    #i is the order of word；a and b are the start and end position of the word; 
    #find the position of words appear in selected text (chars=1)
        toks = []
        for i,(a,b) in enumerate(offsets):
            sm = np.sum(chars[a:b])
            if sm>0: toks.append(i) #characters at a:b appears in selected text 

        #inputs
        s_tok = sentiment_id[train_train.loc[k,'sentiment']]
        input_ids[k,:len(enc.ids)+5] = [0] + enc.ids + [2,2] + [s_tok] + [2] #concatenate text and sentiment as input
        attention_mask[k,:len(enc.ids)+5] = 1
        #outputs:
        if len(toks)>0:
            
            outputs[k,0]=toks[0]+1
            outputs[k,1]=toks[-1]+1
            
            #start_tokens[k,toks[0]+1] = 1 #+1 because we add [0] at the beginning
            #end_tokens[k,toks[-1]+1] = 1
            
            #token_type_ids[k,(toks[0]+1):(toks[-1]+1)]=1
            #start_position.append(toks[0]+1)
            #end_position.append(toks[-1]+1)
    return input_ids,attention_mask,outputs       


# In[ ]:


# Train input
#inputs in different formats
input_ids,attention_mask,outputs=get_encoded_train(train_train)
inputs=np.concatenate([input_ids,attention_mask],axis=1)
# start and end position
start_position=[]
end_position=[]
for k in range(train_train.shape[0]):
    start_position.append(np.where(start_tokens[k] == 1)[0][0])
    end_position.append(np.where(end_tokens[k] == 1)[0][0])


# In[18]:


#Train output
outputs=pd.DataFrame(list(zip(start_position,end_position)),columns=['start','end'])


# Test input

# In[27]:


def get_encoded_test(test_train):
    ct_t = test_train.shape[0]
    input_ids_t = np.ones((ct_t,MAX_LEN),dtype='int32')
    attention_mask_t = np.zeros((ct_t,MAX_LEN),dtype='int32')
    #token_type_ids_t = np.zeros((ct_t,MAX_LEN),dtype='int32')

    for k in range(test_train.shape[0]):

        # INPUT_IDS
        text1 = " "+" ".join(test_train.loc[k,'text'].split())
        enc = tokenizer.encode(text1)                
        s_tok = sentiment_id[test_train.loc[k,'sentiment']]
        input_ids_t[k,:len(enc.ids)+5] = [0] + enc.ids + [2,2] + [s_tok] + [2]
        attention_mask_t[k,:len(enc.ids)+5] = 1
    return input_ids_t,attention_mask_t


# In[ ]:


# Test inputs in different formats
input_ids_t,attention_mask_t=get_encoded_test(test_train)
inputs_t=np.concatenate([input_ids_t,attention_mask_t],axis=1)


# #### In terms of sentiment

# Train input and output

# In[25]:


trainlist={'train_pos':train_pos,'train_neu':train_neu,'train_neg':train_neg}
testlist={'test_pos':test_pos,'test_neu':test_neu,'test_neg':test_neg}


# In[26]:


inputs_s={}
outputs_s={}
for data in trainlist.keys():
    input_ids,attention_mask,outputs_s[data]=get_encoded_train(trainlist[data])
    inputs_s[data]=np.concatenate([input_ids,attention_mask],axis=1)


# In[26]:


inputs_s


# Test input

# In[28]:


inputs_s_t={}
for data in testlist.keys():
    input_ids,attention_mask=get_encoded_test(testlist[data])
    inputs_s_t[data]=np.concatenate([input_ids,attention_mask],axis=1)


# In[100]:


inputs_s_t


# # Part 4 Best subset model

# ## 4.1 Sentiment Score

# ### 1. External Based Lexicon 

# #### Method 1: AFINN Lexicon 

# In[19]:


get_ipython().system('{sys.executable} -m pip install afinn')
from afinn import Afinn


# In[20]:


af = Afinn()

# compute sentiment scores (polarity) and labels
sentiment_scores = [af.score(article) for article in df_train["text"]]


# In[24]:


# get text with the highest polarity and lowest length for POSITIVE data
def highest_polarity_text(corpus_list):
    now_polarity = -100
    highest_text = ""
    highest_text_list = []
    for i in corpus_list:
        if af.score(i) > now_polarity:
            now_polarity = af.score(i)
            highest_text = i
    for i in corpus_list:
        if af.score(i) == now_polarity:
            highest_text_list.extend([i])
    #print(highest_text_list)
    
    now_length = 1000
    shortest_text = ""
    for i in highest_text_list:
        if len(i) < now_length:
            now_length = len(i)
            shortest_text = i
    return shortest_text


# In[25]:


# get text with the lowest polarity and lowest length for NEGATIVE data
def lowest_polarity_text(corpus_list):
    now_polarity = 100
    lowest_text = ""
    lowest_text_list = []
    for i in corpus_list:
        if af.score(i) < now_polarity:
            now_polarity = af.score(i)
            lowest_text = i
    for i in corpus_list:
        if af.score(i) == now_polarity:
            lowest_text_list.extend([i])
    #print(lowest_text_list)
    
    now_length = 1000
    shortest_text = ""
    for i in lowest_text_list:
        if len(i) < now_length:
            now_length = len(i)
            shortest_text = i
    return shortest_text


# In[26]:


#case
corpus = "I'm super happy today"
highest_polarity_text(get_subset_list(corpus))
corpus = "I'm sad today"
lowest_polarity_text(get_subset_list(corpus))


# #### Method 2: SentiWordNet 

# In[29]:


# remove punctuation and lowercase all the words
import nltk
from nltk.corpus import sentiwordnet as swn
import string
non_punct=[]
for s in df_train["text"]:
    s = s.translate(str.maketrans('', '', string.punctuation)).lower()
    non_punct.append(s)


# In[98]:


def compute_score(data):
    sentences = [nltk.sent_tokenize(doc) for doc in data]
    
    stokens=[]
    for i in range(len(sentences)):
        for sent in sentences[i]:
            word = nltk.word_tokenize(sent)
            stokens.append(word)
    
    taggedlist=[]
    for stoken in stokens:        
         taggedlist.append(nltk.pos_tag(stoken))  
    wnl = nltk.WordNetLemmatizer()
    
    score_list=[]
    for idx,taggedsent in enumerate(taggedlist):
        score_list.append([])
        for idx2,t in enumerate(taggedsent):
            newtag=''
            lemmatized=wnl.lemmatize(t[0])
            if t[1].startswith('NN'):
                newtag='n'
            elif t[1].startswith('JJ'):
                newtag='a'
            elif t[1].startswith('V'):
                newtag='v'
            elif t[1].startswith('R'):
                newtag='r'
            else:
                newtag=''       
            if(newtag!=''):    
                synsets = list(swn.senti_synsets(lemmatized, newtag))
                #print(synsets)

                # Getting average of all possible sentiments        
                score=0
                if(len(synsets)>0):
                    for syn in synsets:
                        score+=syn.pos_score()-syn.neg_score()
                    score_list[idx].append(score/len(synsets))
                    #print(score_list)
    if len(score_list)==0:
        return (float(0.0))
    else: 
        leng = len(score_list)
        s=0
        for word_score in score_list:
            s+=sum(word_score)            
            return s / leng


# In[109]:


polarity=[]
p_score=0
for score in non_punct:
    p_score = compute_score(score)
    polarity.append(p_score)
df_train["polarity"]=polarity


# #### Method 3: VADER Lexicon-Based

# In[ ]:


nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()


# In[ ]:


# get text with the highest polarity and lowest length for POSITIVE data
def highest_polarity_text(corpus_list):
    now_polarity = -100
    highest_text = ""
    highest_text_list = []
    for i in corpus_list:
        if analyzer.polarity_scores(i)['compound'] > now_polarity:
            now_polarity = analyzer.polarity_scores(i)['compound']
            highest_text = i
    for i in corpus_list:
        if analyzer.polarity_scores(i)['compound'] == now_polarity:
            highest_text_list.extend([i])
    #print(highest_text_list)
    
    now_length = 1000
    shortest_text = ""
    for i in highest_text_list:
        if len(i) < now_length:
            now_length = len(i)
            shortest_text = i
    return shortest_text


# In[ ]:


# get text with the lowest polarity and lowest length for NEGATIVE data
def lowest_polarity_text(corpus_list):
    now_polarity = 100
    lowest_text = ""
    lowest_text_list = []
    for i in corpus_list:
        if af.score(i) < now_polarity:
            now_polarity = af.score(i)
            lowest_text = i
    for i in corpus_list:
        if af.score(i) == now_polarity:
            lowest_text_list.extend([i])
    #print(lowest_text_list)
    
    now_length = 1000
    shortest_text = ""
    for i in lowest_text_list:
        if len(i) < now_length:
            now_length = len(i)
            shortest_text = i
    return shortest_text


# ### 2. Customized Sentiment Analysis

# #### 1) SVC

# In[35]:


from sklearn import linear_model
from sklearn import metrics

svm = linear_model.SGDClassifier(loss='hinge', random_state = 0) 


# In[38]:


train_train['sentiment_n']=train_train['sentiment'].apply(lambda x: convert_sentiment(x))
test_train['sentiment_n']=test_train['sentiment'].apply(lambda x: convert_sentiment(x))
train_polarity = np.array(train_train['sentiment_n'])
test_polarity = np.array(test_train['sentiment_n'])


# In[ ]:


svm.fit(train_text_matris, train_polarity)
predicted_svm = svm.predict_proba(test_text_matrix) 


# In[132]:


print('Accuracy rate:', np.round(metrics.accuracy_score(test_polarity, predicted_svm), 3))


# #### 2) SVC multi-dimension classifier

# In[48]:


from sklearn import svm


# In[ ]:


#too much time
svc=svm.SVC(decision_function_shape='ovo',probability=True)
svc=svc.fit(train_text_matrix, train_polarity)


# In[ ]:


predict_svc_prob=svc.predict_proba(test_text_matrix)
predict_svc_prob


# #### 3) Random forest classifier

# In[30]:


from sklearn.ensemble import RandomForestClassifier


# In[40]:


clf = RandomForestClassifier(n_estimators=10)
clf = clf.fit(train_text_matrix, train_polarity)


# In[39]:


predict_clf=clf.predict(test_text_matrix)
predict_clf_prob=clf.predict_proba(test_text_matrix)


# #### 4) Substitute VADER

# In[ ]:


#input needs to preprocess: clf predict need to use numerical value; othervies, create customized sentiment vocabulary
def highest_polarity_text(corpus_list):
    now_polarity = -100
    highest_text = ""
    highest_text_list = []
    sent_score={}
    for text in corpus_list:
        total_clf={'pos':0,'neg':0,'neu':0}
        for i in text:
            total_clf['pos']+=clf.predict_proba(i)[2]
            total_clf['neg']+=clf.predict_proba(i)[0]*(-1)
        sent_score[text]=sum(total_clf)/len(text)
    max_score=max(sent_score.items())[1]
    highest_text_list=[key for key in sent_score.keys() if sent_score[key]==max_score]
    
    now_length = 1000
    shortest_text = ""
    for i in highest_text_list:
        if len(i.split()) < now_length:
            now_length = len(i)
            shortest_text = i
    return shortest_text


# ## 4.2 Best subset selection

# ### 1. Select best subset

# #### 1) Positive Strategy

# In[46]:


# get text with polarity > threshold and lowest length >= 1 for POSITIVE data
def select_text_positive(corpus_list, threshold=0):
    potential_text_list = []
    for i in corpus_list:
        if analyzer.polarity_scores(i)['compound'] > threshold:
            potential_text_list.extend([i])
    
    now_length = 1000
    shortest_text = ""
    for i in potential_text_list:
        if (len(i.split(" ")) < now_length) & (len(i.split(" ")) >= 1):
            now_length = len(i.split(" "))
            shortest_text = i
    return shortest_text


# In[47]:


def try_threshold_for_accuracy_positive(threshold_for_pos):
    df_positive["my_selected_text"] = ""
    for index,row in df_positive.iterrows():
        df_positive["my_selected_text"][index] = select_text_positive(get_corpus_list(row[1]), threshold = threshold_for_pos)
    
    jaccard_list = []
    for index,row in df_positive.iterrows():
        jaccard_list.extend([jaccard(row[2], row[4])])
    accuracy = sum(jaccard_list)/len(jaccard_list)
    return(accuracy) 


# In[48]:


thresholds = np.linspace(0,1,20)
acc_rates = [try_threshold_for_accuracy_positive(threshold) for threshold in thresholds]


# In[49]:


plt.plot(thresholds, acc_rates)
plt.xlabel("Threshold score for positive data")
plt.ylabel("Accuracy rate")
plt.title("Accuracy Rate v.s. Threshold for Positive Data")
plt.show()


# In[50]:


best_threshold=thresholds[acc_rates.index(max(acc_rates))]
print("Threshold:",best_threshold, " Accuracy Rate:", max(acc_rates))


# In[56]:


def select_text_positive_final(corpus_list,best_threshold):
    potential_text_list = []
    for i in corpus_list:
        if analyzer.polarity_scores(i)['compound'] > best_threshold:
            potential_text_list.extend([i])
    
    now_length = 1000
    shortest_text = ""
    for i in potential_text_list:
        if (len(i.split(" ")) < now_length) & (len(i.split(" ")) >= 1):
            now_length = len(i.split(" "))
            shortest_text = i
    return shortest_text


# #### 2) Negative Strategy

# In[57]:


# get text with polarity < threshold and lowest length >= 1 for NEGATIVE data
def select_text_negative_1(corpus_list, threshold=0):
    potential_text_list = []
    for i in corpus_list:
        if analyzer.polarity_scores(i)['compound'] < threshold:
            potential_text_list.extend([i])
    
    now_length = 1000
    shortest_text = ""
    for i in potential_text_list:
        if (len(i.split(" ")) < now_length) & (len(i.split(" ")) >= 1):
            now_length = len(i.split(" "))
            shortest_text = i
    return shortest_text


# In[58]:


def try_threshold_for_accuracy_negative(threshold_for_neg):
    df_negative["my_selected_text"] = ""
    for index,row in df_negative.iterrows():
        df_negative["my_selected_text"][index] = select_text_negative_1(get_corpus_list(row[1]), threshold = threshold_for_neg)
    
    jaccard_list = []
    for index,row in df_negative.iterrows():
        jaccard_list.extend([jaccard(row[2], row[4])])
    accuracy = sum(jaccard_list)/len(jaccard_list)
    return(accuracy) 


# In[59]:


thresholds = np.linspace(-1,0,20)
acc_rates = [try_threshold_for_accuracy_negative(threshold) for threshold in thresholds]


# In[60]:


plt.plot(thresholds, acc_rates)
plt.xlabel("Threshold score for negative data")
plt.ylabel("Accuracy rate")
plt.title("Accuracy Rate v.s. Threshold for Negative Data")
plt.show()


# In[61]:


best_threshold=thresholds[acc_rates.index(max(acc_rates))]
print("Threshold:",best_threshold, " Accuracy Rate:", max(acc_rates))


# In[62]:


def select_text_negative_final(corpus_list,best_threshold):
    potential_text_list = []
    for i in corpus_list:
        if analyzer.polarity_scores(i)['compound'] < best_threshold:
            potential_text_list.extend([i])
    
    now_length = 1000
    shortest_text = ""
    for i in potential_text_list:
        if (len(i.split(" ")) < now_length) & (len(i.split(" ")) >= 1):
            now_length = len(i.split(" "))
            shortest_text = i
    return shortest_text


# #### 3) Neutral Strategy: use text as selected_text

# #### Apply functions to the df_train to get my_select_text

# In[63]:


df_train["my_select_text"] = ""


# In[64]:


for index,row in df_train.iterrows():
    if row[3] == "neutral":
        df_train["my_select_text"][index] = df_train["text"][index]
    elif row[3] == "positive":
        if len(row[1].split(" "))<=3:
            df_train["my_select_text"][index] = df_train["text"][index]
        else:
            df_train["my_select_text"][index] = select_text_positive_final(get_corpus_list(df_train["text"][index]))
    elif row[3] == "negative":
        if len(row[1].split(" "))<=3:
            df_train["my_select_text"][index] = df_train["text"][index]
        else:
            df_train["my_select_text"][index] = select_text_negative_final(get_corpus_list(df_train["text"][index]))


# # Part 5 Machine learning model

# ## 5.1 Linear regression multi-dimensional model

# In[59]:


from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression


# 1) Use all data

# In[60]:


lg=LinearRegression().fit(inputs, outputs)
predict_lg=lg.predict(inputs_t)


# Predict score

# In[71]:


def get_my_select(predict,input_ids_t):
    my_selected = []
    for k in range(input_ids_t.shape[0]):
        a = int(predict[k][0])
        b = int(predict[k][1])
        if a>b: 
            st = test_train.loc[k,'text']
        else:
            text1 = " "+" ".join(test_train.loc[k,'text'].split())
            enc = tokenizer.encode(text1)
            st = tokenizer.decode(enc.ids[a-1:b])
        my_selected.append(st)
    return my_selected


# In[73]:


test_train['my_select_lg']=get_my_select(predict_lg,input_ids_t)


# In[74]:


lg_score=test_train.apply(lambda x: jaccard(x["selected_text"], x["my_select_lg"]),axis=1).mean()
print(lg_score)


# 2) Seperate data by semtiments

# In[ ]:


trainlist={'train_pos':train_pos,'train_neu':train_neu,'train_neg':train_neg}
testlist={'test_pos':test_pos,'test_neu':test_neu,'test_neg':test_neg}
datalist=['train_pos']
#train data
inputs_s
outputs_s
#test data
inputs_s_t


# In[101]:


#pos
train_name='train_pos'
test_name='test_pos'
lg_pos=LinearRegression().fit(inputs_s[train_name], outputs_s[train_name])
predict_lg_pos=lg_pos.predict(inputs_s_t[test_name])
test_pos['my_select_lg']=get_my_select(predict_lg_pos,inputs_s_t[test_name])

lg_pos_score=test_pos.apply(lambda x: jaccard(x["selected_text"], x["my_select_lg"]),axis=1)


# In[102]:


#neu
train_name='train_neu'
test_name='test_neu'
lg_neu=LinearRegression().fit(inputs_s[train_name], outputs_s[train_name])
predict_lg_neu=lg_neu.predict(inputs_s_t[test_name])
test_neu['my_select_lg']=get_my_select(predict_lg_neu,inputs_s_t[test_name])

lg_neu_score=test_neu.apply(lambda x: jaccard(x["selected_text"], x["my_select_lg"]),axis=1)


# In[103]:


#neg
train_name='train_neg'
test_name='test_neg'
lg_neg=LinearRegression().fit(inputs_s[train_name], outputs_s[train_name])
predict_lg_neg=lg_neg.predict(inputs_s_t[test_name])
test_neg['my_select_lg']=get_my_select(predict_lg_neg,inputs_s_t[test_name])

lg_neg_score=test_neg.apply(lambda x: jaccard(x["selected_text"], x["my_select_lg"]),axis=1)


# In[104]:


lg_s_score=(sum(lg_pos_score)+sum(lg_neu_score)+sum(lg_neg_score))/(len(lg_pos_score)+len(lg_neu_score)+len(lg_neg_score))
print(lg_s_score)


# ## 5.2 SVR multi-dimensional model

# In[29]:


from sklearn.datasets import make_regression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import LinearSVR


# ### 1. Use all data

# In[115]:


model = LinearSVR()
wrapper = MultiOutputRegressor(model)
svr=wrapper.fit(inputs, outputs)
predict_svr=svr.predict(inputs_t)


# In[118]:


test_train['my_select_svr']=get_my_select(predict_svr,input_ids_t)
svr_score=test_train.apply(lambda x: jaccard(x["selected_text"], x["my_select_lg"]),axis=1).mean()
print(svr_score)


# ### 2. Separately predict start position and end position  
# 1) predict the start position

# In[ ]:


from sklearn import linear_model

svm = linear_model.SGDClassifier(loss='hinge', random_state = 0) 
svm.fit(inputs,start_position)
predicted_svm = svm.predict(inputs_t) 


# 2) predict the end position

# In[26]:


svm.fit(inputs,end_position)
predicted_end_svm = svm.predict(inputs_t) 


# In[29]:


my_selected = []
for k in range(input_ids_t.shape[0]):
    a = predicted_svm[k]
    b = predicted_end_svm[k]
    if a>b: 
        st = test_train.loc[k,'text']
    else:
        text1 = " "+" ".join(test_train.loc[k,'text'].split())
        enc = tokenizer.encode(text1)
        st = tokenizer.decode(enc.ids[a-1:b])
    my_selected.append(st)
test_train["my_select_text"]=my_selected


# ## 5.3 Random Forest multi-dimensional model

# In[108]:


from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor


# In[109]:


rf = RandomForestRegressor().fit(inputs, outputs)
predict_rf=rf.predict(inputs_t)
test_train['my_select_rf']=get_my_select(predict_rf,input_ids_t)
rf_score=test_train.apply(lambda x: jaccard(x["selected_text"], x["my_select_lg"]),axis=1).mean()
print(rf_score)


# # Part 6 Deep Learning:  Tensorflow roBERTa

# In[ ]:


#packages needed

import sys

get_ipython().system('{sys.executable} -m pip install numpy')
import numpy as np 

get_ipython().system('{sys.executable} -m pip install pandas')
import pandas as pd

get_ipython().system('{sys.executable} -m pip install --upgrade tensorflow')
import tensorflow as tf

get_ipython().system('{sys.executable} -m pip install keras')
import tensorflow.keras.backend as K
from sklearn.model_selection import StratifiedKFold
from transformers import *

get_ipython().system('{sys.executable} -m pip install tokenizers')
import tokenizers


get_ipython().system('{sys.executable} -m pip install torch torchvision')
import torch


# In[23]:


def scheduler(epoch):
    return 3e-5 * 0.2**epoch


# In[27]:


def build_model():
    
    # Initialize keras layers
    ids = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    att = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
    tok = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)

    # Fetching pretrained models 
    
    config = RobertaConfig.from_pretrained('config-roberta-base.json')
    bert_model = TFRobertaModel.from_pretrained('pretrained-roberta-base.h5',config=config)
    x = bert_model(ids,attention_mask=att,token_type_ids=tok)
    
    # Setting up layers
    x1 = tf.keras.layers.Dropout(0.1)(x[0]) 
    x1 = tf.keras.layers.Conv1D(128, 2,padding='same')(x1)
    x1 = tf.keras.layers.LeakyReLU()(x1)
    x1 = tf.keras.layers.Conv1D(64, 2,padding='same')(x1)
    x1 = tf.keras.layers.Dense(1)(x1)
    x1 = tf.keras.layers.Flatten()(x1)
    x1 = tf.keras.layers.Activation('softmax')(x1)
    
    x2 = tf.keras.layers.Dropout(0.1)(x[0]) 
    x2 = tf.keras.layers.Conv1D(128, 2, padding='same')(x2)
    x2 = tf.keras.layers.LeakyReLU()(x2)
    x2 = tf.keras.layers.Conv1D(64, 2, padding='same')(x2)
    x2 = tf.keras.layers.Dense(1)(x2)
    x2 = tf.keras.layers.Flatten()(x2)
    x2 = tf.keras.layers.Activation('softmax')(x2)

    # Initializing input,output for model.THis will be trained in next code
    model = tf.keras.models.Model(inputs=[ids, att, tok], outputs=[x1,x2])
    
    #Adam optimizer for stochastic gradient descent. if you are unware of it - https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
    model.compile(loss='binary_crossentropy', optimizer=optimizer)

    return model


# In[31]:


preds_start = np.zeros((input_ids_t.shape[0],MAX_LEN))
preds_end = np.zeros((input_ids_t.shape[0],MAX_LEN))
n_splits=5

DISPLAY=1
for i in range(5):
    print('#'*25)
    print('### MODEL %i'%(i+1))
    print('#'*25)
    
    K.clear_session()
    model = build_model()
    # Pretrained model
    model.load_weights('v4-roberta-%i.h5'%i)

    print('Predicting Test...')
    preds = model.predict([input_ids_t,attention_mask_t,token_type_ids_t],verbose=DISPLAY)
    preds_start += preds[0]/n_splits
    preds_end += preds[1]/n_splits


# In[34]:


all = []
for k in range(input_ids_t.shape[0]):
    # Argmax - Returns the indices of the maximum values along axis
    a = np.argmax(preds_start[k,])
    b = np.argmax(preds_end[k,])
    if a>b: 
        st = df_test.loc[k,'text']
    else:
        text1 = " "+" ".join(df_test.loc[k,'text'].split())
        enc = tokenizer.encode(text1)
        st = tokenizer.decode(enc.ids[a-1:b])
    all.append(st)


# In[35]:


df_test['selected_text'] = all
submission=df_test[['textID','selected_text']]
submission.to_csv('submission.csv',index=False)
submission.head(5)


# In[ ]:





# In[ ]:




