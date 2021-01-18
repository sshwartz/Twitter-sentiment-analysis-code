from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import word_tokenize
from common_functions import convert_label_to_value, SentimentScores, preprocess_one_line
import pandas as pd


class WordBasedScorer(object):
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = stopwords.words('english')
        neg_file = open('/home/data/negative-words.txt', 'r')
        negatives_raw = neg_file.readlines()
        self.negatives = []
        for n in negatives_raw:
            neg = n.rstrip()
            self.negatives.append(neg)

        pos_file = open('/home/data/positive-words.txt', 'r')
        positives_raw = pos_file.readlines()
        self.positives = []
        for p in positives_raw:
            pos = p.rstrip()
            self.positives.append(pos)

    def compute_score(self, review):
        score = 0
        review = preprocess_one_line(review, self.lemmatizer, True, self.stop_words)
        toks = word_tokenize(review)
        for s in toks:
            if s in self.positives:
                score += 1
            if s in self.negatives:
                score += -1
        if score == 0:
            computed_score = 0
        elif score < 0:
            computed_score = -1
        else:
            computed_score = 1
        return computed_score

def analyze_sentiment():
    scores = SentimentScores('word-based')
    scorer = WordBasedScorer()
    df = pd.read_csv('/home/data/tweets_preprocessed.txt', names=['sentiment', 'review'])

    # Randomly take the first 1000
    df = df.sample(frac=1)  #shuffle the rows
    df = df.head(1000)  # take the first 1000
    for index, row in df.iterrows():
        rev = row['review']
        gt_label = row['sentiment']
        gt = convert_label_to_value(gt_label)
        computed_score = scorer.compute_score(rev)
        scores.update(computed_score, gt)
    outfile = scores.output_results()
    outfile.close()

analyze_sentiment('tweets')


