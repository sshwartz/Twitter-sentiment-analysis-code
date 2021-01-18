# Create input files for third party sentiment analyzers including AWS, GCS, CoreNLP
import pandas as pd
from sklearn.model_selection import train_test_split
import csv

class CreateTestFilesNoPreprocessing(object):

    def __init__(self):
        rev_file = open('/home/data/tweets.txt', 'r')
        self.review_list = rev_file.readlines()

    def preprocess(self):
        self.review_list.pop(0)  # remove the header
        test_file = open('/home/data/temp_tweets.txt', 'w')
        for r in self.review_list:
            # Remove trailing newline characters
            rev_clean = r.rstrip()
            rsplit = rev_clean.split(',')

            review = rsplit[1]
            rec = rsplit[0].upper() + ", " + review + "\n"
            test_file.write(rec)
        test_file.close()


    def create_test_data(self):
        df = pd.read_csv('/home/data/temp_tweets.txt', names=['sentiment', 'review'])
        # Filter out bad data
        df_positive = df[df.sentiment == 'POSITIVE']
        df_negative = df[df.sentiment == 'NEGATIVE']
        df_neutral = df[df.sentiment == 'NEUTRAL']
        df = pd.concat([df_positive, df_negative, df_neutral])

        # Write out data with a comma separating the sentiment from the review for direct classifiers like AWS and CoreNLP
        df_train, df_test = train_test_split(df, test_size=.3, shuffle=True)
        df_test.to_csv('/home/data/tweets_no_preprocessing.txt', sep=',', header=True, index=False, quoting=csv.QUOTE_MINIMAL, escapechar=' ')


cr = CreateTestFilesNoPreprocessing()
cr.preprocess()
cr.create_test_data()




