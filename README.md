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
1. A quick look at the sentiment:  
    - Negative: 7781
    - Neural: 11117
    - Positive: 8582
2. Missing data. 
  Find all missing data and fill it with NaN.  
  ```python
  null_train_data=df_train[df_train.isnull().any(axis=1)]
  df_train=df_train.fillna('')
  ```  
3. Length of text and selected text  
    ![difference of length]()  
    ![average difference of length]()  
    Most selected text of neural tweet has the same length as the original text. Negative and positive selected text are very different from the original text.  
evaluate predict accuracy:
Jaccard similarity for string. 
```python
def jaccard(str1, str2): 
    a = set(str1.lower().split()) 
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))
```
