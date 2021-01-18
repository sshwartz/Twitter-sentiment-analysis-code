# Create input files for supervised learning sentiment classifiers
# Preprocessing includes lemmatization

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import pandas as pd
from sklearn.model_selection import train_test_split
import csv
from SentimentAnalysis.common_functions import preprocess_one_line, get_review_text, maybe_print_count


class CreateTrainTestFile(object):

    def __init__(self, review_type):
        self.lemmatize_flag = True
        self.lemmatizer = WordNetLemmatizer()
        rev_file = open('/home/data/tweets.txt', 'r')
        self.review_list = rev_file.readlines()
        self.stop_words = stopwords.words('english')

    def preprocess(self):
        self.review_list.pop(0)  # remove the header
        train_test_file = open('/home/data/temp_sl_tweets.txt', 'w')
        for r in self.review_list:
            # Remove trailing newline characters
            rev_clean = r.rstrip()
            rsplit = rev_clean.split(',')

            review = rsplit[1]
            review = preprocess_one_line(review, self.lemmatizer, self.lemmatize_flag, self.stop_words)
            rec = rsplit[0].upper() + ", " + review + "\n"
            train_test_file.write(rec)
        train_test_file.close()


    def output_files(self):
        df = pd.read_csv('/home/data/temp_sl_tweets.txt', names=['sentiment', 'review'])
        # Remove junk
        df_positive = df[df.sentiment == 'POSITIVE']
        df_negative = df[df.sentiment == 'NEGATIVE']
        df_neutral = df[df.sentiment == 'NEUTRAL']
        df = pd.concat([df_positive, df_negative, df_neutral])

        # Write out data with a comma separating the sentiment from the review for supervised learning classifiers other than Fastext
        df.to_csv('/home/data/tweets_preprocessed.txt', header=True, index=False) # Write out summary file with commas

        # Write out data for fastext and word count baseline with no commma and separate train and test files
        df['sentiment'].replace('POSITIVE', '__label__POSITIVE', inplace=True)
        df['sentiment'].replace('NEGATIVE', '__label__NEGATIVE', inplace=True)
        df['sentiment'].replace('NEUTRAL', '__label__NEUTRAL', inplace=True)
        di = dict()
        di['POSITIVE'] = '__label__POSITIVE'
        di['NEGATIVE'] = '__label__NEGATIVE'
        di['NEUTRAL'] = '__label__NEUTRAL'
        df.replace({"sentiment": di}, inplace=True)
        df_train, df_test = train_test_split(df, test_size=.3, shuffle=True)
        df_train.to_csv('/home/data/fastext_train_tweets.txt', sep=' ', header=False, index=False, quoting=csv.QUOTE_NONE, escapechar=' ')
        df_test.to_csv('/home/data/fastext_test_tweets.txt', sep=' ', header=False, index=False, quoting=csv.QUOTE_NONE, escapechar=' ')

        # Write out data for DistilBERT
        di = dict()
        di["__label__POSITIVE"] = 1
        di["__label__NEGATIVE"] = -1
        di["__label__NEUTRAL"] = 0
        df.replace({"sentiment": di}, inplace=True)
        df.to_csv('/home/data/bert_tweets.tsv', sep='\t', header=False, index=False, columns=["review", "sentiment"],
                      quoting=csv.QUOTE_NONE, escapechar=' ')

cr = CreateTrainTestFile()
cr.preprocess()
cr.output_files()



