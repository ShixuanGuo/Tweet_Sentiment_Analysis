# Tweets_Sentiment_Analysis
Predict phases from tweet to support sentiment analysis

## Part 1 Project Introduction  
1. **Problem description**  
  In this project, I conducted exploratory data analysis for tweets and built different models to predict the part of tweet that reflects the labeled sentiment.   
2. **Data description**  
  This project use tweet data extracted from [Figure Eight's Data for Everyone platform](https://appen.com/resources/datasets/). The dataset is titled Sentiment Analysis: Emotion in Text tweets with existing sentiment labels, used here under creative commons attribution 4.0. international licence.  
  The file names 'tweet_train.csv', containing
    - text of a tweet
    - pre-selected phrase
    - sentiment
3. **Implemented Algorithms**
    - Exploratory data analysis (EDA)  
    - Prediction Model:  
        - VADER Lexicon model 
        - Customized Lexicon score
        - Machine learning  
4. **Packages**  
    - numpy
    - pandas
    - nltk
    - matplotlib
    - sklearn


## Part 2 Exploratory Data Analysis  
1. A quick look at the **sentiment**:  
    - Negative: 7781
    - Neural: 11117
    - Positive: 8582
2. **Missing data**  
    Find all missing data and fill it with NaN.  
    ```python
    null_train_data=df_train[df_train.isnull().any(axis=1)]
    df_train=df_train.fillna('')
    ```  
3. **Length of text and selected text**  
    ![difference of length]()  
    ![average difference of length]()  
    - Neural: Most selected text has the same length as the original text. Average length of selected text is 12.  
    - Negative and positive: selected text are very different from the original text. The average length of selected text is around 3.5-4.  
    Thus, split data based on sentiment and analyze them separately. Have a deeper look at neural tweets:  
    ![neural describe]()  

4. **URL**  
    1) Find all URLs in text and selected_text  
    `df_train['url']=df_train['text'].str.lower().apply(lambda x: find_link(x))`.  
    2) Compare URLs in original text and selected text
    ![url]()  
    - Only 3 urls in negative and positive text are selected. We can remove urls in negative and positive text.  
    
5. **Punctuation**  
    1) Find all punctuations and count the number of punctuations in text and selected_text   
    ```python
    df_train['text'].apply(lambda x: find_punctuation(x))
    df_train['text'].apply(lambda x:len(find_punctuation(x)))
    ```  
    2) Identify difference between text and selected text  
    `df_train.apply(lambda x: find_difference(x['text_punctuation'],x['target_punctuation']), axis=1)`  
    3) Analysis insights  
    - 11013 of 27480 selected text (nearly a half) do not contain any punctuations.  
    - The most frequently occurring punctuation remained in selected text are .!`,*.  
    ![most frequent punctuation]()  
    - 15718 selected text (over a half) do not have different punctuations with text. Others usually have one punctuation different from text.  The most frequently occurring different punctuation between text and select text is ".".
    ![different punctuation]()  
    4) Analysis in terms of sentiment  
    ![no punctuation]()  
    - Natural texts always keep the original punctuations; positive and negative texts' punctuations changed more.  
    ![popular punctuation]()
    - Some punctuations have interesting performances in different sentiment:  
    * always occurs at negative selected text; ! always occurs at positive selected text.  
    
evaluate predict accuracy:
Jaccard similarity for string. 
```python
def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))
```
