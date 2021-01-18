# Twitter-sentiment-analysis-code

This repository holds the source code for the blog post [12 Twitter Sentiment Analysis Algorithms Compared](https://www.aiperspectives.com/twitter-sentiment-analysis).

These are the source code files in the repository:

### create_pos_neg_word_files.py
Reads [this list](https://www.kaggle.com/nltkdata/opinion-lexicon) of positive and negative sentiment words, performs the lemmatization and other transformations discussed in the post, and outputs a file of positive sentiment words and one of negative sentiment words that are used by word_baseline_classifier.py.

### create_third_party_sentiment_files.py
Reads in the tweet data and outputs files are used by third_party_sentiment_classifiers.py.

### create_sl_datasets.py
Reads in the tweet data, preprocesses the tweets as discussed in the blog post, and outputs files that are used by sl_classifiers_sentiment.py.

### word_baseline_classifier.py
Performs sentiment analysis based on comparing positive/negative words from the sentiment lexicon to words in the tweets.

### third_party_sentiment_classifiers.py
Performs sentiment analysis using Amazon Comprehend, Google Cloud Services, and Stanford CoreNLP.

### sl_classifiers_sentiment.py
Performs sentiment analysis using Naive Bayes, Linear SVC, XGBoost, Decision Trees, k-Nearest Neighbors, and a Keras-based deep learning algorithm.

### fastext_sentiment.py
Performs sentiment analysis using the Facebook fasText algorithm.

### common_functions.py
Functions and classes used by more than one of the above Python scripts.

### Distilbert sentiment analysis.ipynb
Performs sentiment analysis by taking a pre-trained DistilBERT model and feeding the features into a logistic regression classifier.  This file is a Google Colab notebook that can be run on Google Colab (but requires the Pro option for memory).

The following data files are used by the above programs and are assumed to be in the /home/data folder:

### positive-words-raw.txt
A list of positive sentiment words downloaded from [Kaggle](https://www.kaggle.com/nltkdata/opinion-lexicon).

### negative-words-raw.txt
A list of positive negative sentiment words downloaded from [Kaggle](https://www.kaggle.com/nltkdata/opinion-lexicon).

### tweets.txt
A list of tweets downloaded from [Kaggle](https://www.kaggle.com/crowdflower/twitter-airline-sentiment).

