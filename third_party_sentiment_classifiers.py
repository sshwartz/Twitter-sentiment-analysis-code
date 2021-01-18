from common_functions import convert_label_to_value, SentimentScores, maybe_print_count, preprocess_one_line
from pycorenlp import StanfordCoreNLP
from google.cloud import language_v1
import boto3
import pandas as pd


class GCSScorer(object):
    def __init__(self):
        self.client = language_v1.LanguageServiceClient()

    def compute_score(self, rev):
        document = language_v1.Document(content=rev, type_=language_v1.Document.Type.PLAIN_TEXT)
        sentiment = self.client.analyze_sentiment(request={'document': document}).document_sentiment
        score = sentiment.score
        if score > .33:
            return 1
        elif score < -.33:
            return -1
        else:
            return 0


class AWSScorer(object):
    def __init__(self):
        self.comprehend = boto3.client(service_name='comprehend', region_name='us-east-1')

    def compute_score(self, rev):
        try:
            s = self.comprehend.detect_sentiment(Text=rev, LanguageCode='en')
        except Exception as e:
            print(e, rev)
            return 0
        sentiment = s['Sentiment']
        if sentiment == 'POSITIVE':
            return 1
        elif sentiment == 'NEGATIVE':
            return -1
        else:
            return 0


class CoreNLPScorer(object):
    def __init__(self):
        self.cnlp = StanfordCoreNLP('http://localhost:9000')

    def compute_score(self, rev):
        results = self.cnlp.annotate(rev, properties={
            'annotators': 'sentiment',
            'parse.maxlen': 70,
            'outputFormat': 'json',
            'timeout': 50000,
        })
        try:
            s = results["sentences"][0]["sentiment"]
        except Exception as e:
            return 0
        if s == 'Neutral':
            return 0
        elif s == 'Positive':
            return 1
        else:
            return -1


def analyze_sentiment(tool_name):
    if tool_name == 'core_nlp':
        scorer = CoreNLPScorer()
    elif tool_name == 'aws':
        scorer = AWSScorer()
    elif tool_name == 'google':
        scorer = GCSScorer()
    else:
        return
    scores = SentimentScores(str(scorer))
    df = pd.read_csv('/home/data/tweets_no_preprocessing.txt', names=['sentiment', 'review'])

    # Randomly take the first 1000
    df = df.sample(frac=1)  #shuffle the rows
    df = df.head(1000)  # take the first 1000
    cnt = 0
    for index, row in df.iterrows():
        rev = row['review']
        gt_label = row['sentiment']
        gt = convert_label_to_value(gt_label)
        computed_score = scorer.compute_score(rev)
        scores.update(computed_score, gt)
        # maybe_print_count(cnt, 50)
        cnt += 1
    outfile = scores.output_results()
    scores.write_string_to_outfile(tool_name + '\n')
    outfile.close()

analyze_sentiment('aws')
analyze_sentiment('core_nlp')
analyze_sentiment('google')
