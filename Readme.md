# Fake News Detection – Project Report

## Methodology:
Starting with loading the dataset, it is split into [Fake News] and [Real News]. First, we observe both separately – their column names, number of rows and columns, etc.

After that, we start Data Preprocessing. We first add a `label` column in each dataset before merging, where `0` indicates fake and `1` indicates real. Then we merge both datasets and shuffle the data randomly. We then inspect the merged dataset using `.info()`, `.shape`, etc.

Next, we uppercase the column names to ensure consistency. We check for duplicates and drop unnecessary columns like `Date` and `Subject` as they add no value.

For cleaning the main `Text` column:
- We strip whitespace, convert to lowercase, and remove special characters and punctuation.
- We tokenize the text using `TF-IDF Vectorizer` and create a `tokens` column.
- We apply lemmatization to reduce each word to its base form, such as "running" to "run".
- For more advanced processing, we apply POS-based lemmatization.

After data cleaning, we begin **Exploratory Data Analysis (EDA)**:
- We visualize the distribution of real vs fake news using a bar plot.
- We create word clouds for both fake and real news.
- We plot the distribution of article lengths.
- We perform sentiment analysis and observe that fake news tends to have lower polarity.

### Feature Extraction:
- We apply `TF-IDF Vectorizer` to the lemmatized text.
- We add two extra features: `text_length` and `sentiment_score`, as they are useful indicators.

### Model Training:
- Features used: [TF-IDF text, text_length, sentiment_score]
- Labels used: `Label`
- Data is split into training and test sets.
- We apply both **Logistic Regression** and **MultinomialNB** models to train and evaluate.

### Evaluation Metrics:
We use accuracy, precision, recall, F1 score, and confusion matrix.

## Design Choice:
- **Logistic Regression** is preferred because it performs well with binary classification problems.
- **MultinomialNB** is used because it’s efficient for text data, especially with bag-of-words or TF-IDF representations.

## Evaluation Results:

### Logistic Regression:
- **Training Accuracy**: 0.9923
- **Test Accuracy**: 0.9866
- **Classification Report**:
```
              precision    recall  f1-score   support
       0         0.99       0.98      0.99      4652
       1         0.98       0.99      0.99      4286

    accuracy                          0.99      8938
   macro avg         0.99       0.99     0.99      8938
weighted avg         0.99       0.99     0.99      8938
```

### MultinomialNB:
- **Training Accuracy**: 0.9486
- **Test Accuracy**: 0.9403
- **Classification Report**:
```
              precision    recall  f1-score   support
       0         0.95       0.93      0.94      4652
       1         0.93       0.95      0.94      4286

    accuracy                          0.94      8938
   macro avg         0.94       0.94     0.94      8938
weighted avg         0.94       0.94     0.94      8938
```

## Key Insights:
- The sentiment analysis graph shows fake news often has lower polarity scores.
- Fake articles are written with more emotionally charged or extreme language compared to real news.