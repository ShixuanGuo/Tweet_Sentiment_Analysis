# Tweets_Sentiment_Analysis
Predict phases from tweet to support sentiment analysis

## Part 1 Project Introduction  
1. **Problem description**  
  In this project, I conducted exploratory data analysis for tweets and built different models to predict the part of tweet that reflects the labeled sentiment. Use jaccard score as predict accuaracy.   
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
    
    <img src="https://github.com/ShixuanGuo/Tweet_Sentiment_Analysis/blob/master/img/difference%20of%20length.png" alt="difference of length" width="403.2" height="254.4">  
      
    - Neural: Most selected text has the same length as the original text. Average length of selected text is 12.  
    - Negative and positive: selected text are very different from the original text. The average length of selected text is around 3.5-4.  
    
    <img src="https://github.com/ShixuanGuo/Tweet_Sentiment_Analysis/blob/master/img/average%20difference%20of%20length.png" alt="average difference of length" width="386.4" height="144">  
      
    Thus, split data based on sentiment and analyze them separately. Have a deeper look at neural tweets:  
      
    <img src="https://github.com/ShixuanGuo/Tweet_Sentiment_Analysis/blob/master/img/neural%20describe.png" alt="neural describe" width="372" height="112.8">   

4. **URL**  
    1) Find all URLs in text and selected_text  
    `df_train['url']=df_train['text'].str.lower().apply(lambda x: find_link(x))`.  
    2) Compare URLs in original text and selected text
      
    <img src="https://github.com/ShixuanGuo/Tweet_Sentiment_Analysis/blob/master/img/url.png" alt="url" width="402" height="80.4">  
    
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
      
    <img src="https://github.com/ShixuanGuo/Tweet_Sentiment_Analysis/blob/master/img/most%20frequently%20puntuation.png" alt="most frequent punctuation" width="190.8" height="294">  
    
    - 15718 selected text (over a half) do not have different punctuations with text. Others usually have one punctuation different from text.  The most frequently occurring different punctuation between text and select text is ".".  
      
    <img src="https://github.com/ShixuanGuo/Tweet_Sentiment_Analysis/blob/master/img/different%20punctuation.png" alt="different punctuation" width="189.6" height="121.2">  
    
    4) Analysis in terms of sentiment  
      
    <img src="https://github.com/ShixuanGuo/Tweet_Sentiment_Analysis/blob/master/img/no%20punctuation.png" alt="no punctuation" width="364.6" height="80.4">  
    
    - Natural texts always keep the original punctuations; positive and negative texts' punctuations changed more.   
    - Some punctuations have interesting performances in different sentiment:  
    * always occurs at negative selected text; ! always occurs at positive selected text.  

6. **Part of Speech**  
    1) Define parts of speech and add tags using nltk package for text and selected text. Then remove noises of tags.  
    2) Classify parts of speech with easy-to-understand expressions.  
    `df_train["text_class"] = df_train["text_tags"].apply(identify_class)`  
    3) Analysis  
    - The proportion of the selected text in text by parts of speech:  
    
    <img src="https://github.com/ShixuanGuo/Tweet_Sentiment_Analysis/blob/master/img/proportion%20by%20parts%20of%20speech.png" alt="proportion by parts of speech" width="222" height="226.5">  
      
    - Parts of speech distribution by sentiments  
    <img src="https://github.com/ShixuanGuo/Tweet_Sentiment_Analysis/blob/master/img/Positive%20parts%20of%20speech.png" alt="positive parts of speech" width="607" height="233">  
    <img src="https://github.com/ShixuanGuo/Tweet_Sentiment_Analysis/blob/master/img/Negative%20parts%20of%20speech.png" alt="negative parts of speech" width="607" height="233">  
    <img src="https://github.com/ShixuanGuo/Tweet_Sentiment_Analysis/blob/master/img/Neutral%20parts%20of%20speech.png" alt="neutral parts of speech" width="607" height="233">  
    
## Part 3 Preprocessing  
1. **Split the data**  
    1) Convert sentiment into -1,0 and 1  
    2) Split data by sentiment  
    3) Split train and test dataset  
2. **Get input**  
    1) Subset of text: use function: `get_subset_list(text)`  
    2) TF-IDF vectorization  
    ```python
    vectorizer = TfidfVectorizer(max_features=1000)  
    train_text_matrix = vectorizer.fit_transform(train_train['text'].values.astype('U')).toarray()
    train_matrix_names = vectorizer.get_feature_names() 
    train_text_matrix_table = pd.DataFrame(np.round(train_text_matrix, 2), columns = train_matrix_names)
    ```  
    Head of train set vectorization matrix:  
    
    <img src="" alt="TF_IDF matrix" width="607" height="233">  
    
    3) Encode start-end position of selected phrase  
    First, initial tokenizer using Roberta database and encode sentiment:  
    ```python
    MAX_LEN = 96
    tokenizer = tokenizers.ByteLevelBPETokenizer(
      vocab_file='vocab-roberta-base.json', 
      merges_file='merges-roberta-base.txt', 
      lowercase=True,
      add_prefix_space=True)
    sentiment_id = {'positive': 1313, 'negative': 2430, 'neutral': 7974}
    ```  
    Second, encode train and test inputs and outputs, and tune the inputs' format.  
    ```python
    #train
    input_ids,attention_mask,outputs=get_encoded_train(train_train)
    inputs=np.concatenate([input_ids,attention_mask],axis=1)
    outputs=pd.DataFrame(list(zip(start_position,end_position)),columns=['start','end'])
    #test
    input_ids_t,attention_mask_t=get_encoded_test(test_train)
    inputs_t=np.concatenate([input_ids_t,attention_mask_t],axis=1)
    ```  
## Part 4 Best Subset Model
1. **Sentiment scoring model**  
    1) VADER Lexicon
    ```python
    analyzer = SentimentIntensityAnalyzer()
    positive=highest_polarity_text(get_subset_list(text))
    negative=lowest_polarity_text(get_subset_list(text))
    ```  
    2) AFINN Lexicon  
    ```python
    af = Afinn()
    ```  
    Problem: the text can only distinguish words such as "sad". Under the shorest-length logic, it would only select the one-word instead of a phrase.  
    
    3) SentiWordNet Lexicon  
     I also tried SentiWordNet and other lexicon bases. They did not perform well in this project.  
   4) Customized sentiment score-SVC score  
    ```python
    svm = linear_model.SGDClassifier(loss='hinge', random_state = 0) 
    svm.fit(train_text_matris, train_polarity)
    predicted_svm = svm.predict_proba(test_text_matrix) 
    ```  
    Substitute lexicon with SVC model inside the `highest_polarity_text(get_subset_list(text))` function.  
2. **Best subset selection**  
    1) Positive text strategy  
    For sentiment score higher than threshold, select the shortest subset. Try different threshold within (0,1) and select the threshold given the highest accuracy.   
    ```python
    thresholds = np.linspace(0,1,20)
    acc_rates = [try_threshold_for_accuracy_positive(threshold) for threshold in thresholds]
    best_threshold=thresholds[acc_rates.index(max(acc_rates))]
    select_text_positive_final(corpus_list,best_threshold)
    ```  
    <img src="" alt="best threshold" width="607" height="233">  
    
    2) Negative text strategy  
    For sentiment score lower than threshold, select the shortest subset. Select best threshold using the same method.  
    ```python
    select_text_negative_final(corpus_list,best_threshold)
    ```  
    3) Neutral text strategy  
    Use text as selected_text. According to our EDA, neutral selected texts are very similar with original texts.  

## Part 5 Machine Learning Model  
1. **Linear regression multidimensional prediction**  
    Predictors are original text and sentiment. Responsers are positions of selected text in original text.  
    1) Fit the model with all data
    ```python
    lg=LinearRegression().fit(inputs, outputs)
    predict_lg=lg.predict(inputs_t)
    test_train['my_select_lg']=get_my_select(predict_lg,input_ids_t)
    ```  
    2) Separately fit the model by sentiment  
2. **SVR multidimensional prediction**  
    1) Replace the lg model with SVR.  
    ```python
    model = LinearSVR()
    wrapper = MultiOutputRegressor(model)
    ```
    2) Try to predict the position as a whole and separately predict start and end position.  
3. **Random forest multidimensional prediction**  
    
## Part 6 Deep Learning: Tensorflow roBERTa  
1. **Train the model**  
    Use cross validation to train the roBERTa model and set fold=5. 
    ```python
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
        model.load_weights('v4-roberta-%i.h5'%i)

        print('Predicting Test...')
        preds = model.predict([input_ids_t,attention_mask_t,token_type_ids_t],verbose=DISPLAY)
        preds_start += preds[0]/n_splits
        preds_end += preds[1]/n_splits
    ```
2. **Make prediction**  
    ```python
    predict_text = []
    for k in range(input_ids_t.shape[0]):
        a = np.argmax(preds_start[k,])
        b = np.argmax(preds_end[k,])
        if a>b: 
            st = df_test.loc[k,'text']
        else:
            text1 = " "+" ".join(df_test.loc[k,'text'].split())
            enc = tokenizer.encode(text1)
            st = tokenizer.decode(enc.ids[a-1:b])
        predict_text.append(st)
    ```  
