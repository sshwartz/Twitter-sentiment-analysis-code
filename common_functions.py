import numpy as np
from scipy.stats import spearmanr
import re
import datetime


class SentimentScores(object):
    def __init__(self, classifier_name):
        self.true_positives = 0
        self.true_negatives = 0
        self.true_neutrals = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.false_neutrals = 0
        self.total = 0
        self.predicted = []
        self.ground_truth = []
        self.start_time = datetime.datetime.now()
        self.outfile = open('/home/data/outfile.txt', 'a')
        self.outfile.write(f'\n\n\nStart Time: {str(self.start_time)}  Classifier: {classifier_name}')

    def update(self, pred, gt):
        self.ground_truth.append(gt)
        self.predicted.append(pred)
        if pred == 1:
            if gt == 1:
                self.true_positives += 1
            else:
                self.false_positives += 1
        elif pred == -1:
            if gt == -1:
                self.true_negatives += 1
            else:
                self.false_negatives += 1
        else:
            if gt == 0:
                self.true_neutrals += 1
            else:
                self.false_neutrals += 1
        self.total += 1

    def output_results(self):
        gt_arr = np.array(self.ground_truth)
        pred_arr = np.array(self.predicted)
        r, p = spearmanr(gt_arr, pred_arr)
        self.outfile.write(f'\nr {r:5.2f}\n')
        true_positive_pct = self.true_positives * 100 / self.total
        true_negative_pct = self.true_negatives * 100 / self.total
        true_neutral_pct = self.true_neutrals * 100 / self.total
        false_positive_pct = self.false_positives * 100 / self.total
        false_negative_pct = self.false_negatives * 100 / self.total
        false_neutral_pct = self.false_neutrals * 100 / self.total
        self.outfile.write(f'True Positives: {true_positive_pct:2.2f}%\n')
        self.outfile.write(f'True Negatives: {true_negative_pct:2.2f}%\n')
        self.outfile.write(f'True Neutral: {true_neutral_pct:2.2f}%\n')
        self.outfile.write(f'False Positives: {false_positive_pct:2.2f}%\n')
        self.outfile.write(f'False Negatives: {false_negative_pct:2.2f}%\n')
        self.outfile.write(f'False Neutrals: {false_neutral_pct:2.2f}%\n')
        total_time = datetime.datetime.now() - self.start_time
        self.outfile.write(f'Runtime: {str(total_time)}\n')
        return self.outfile

    def write_string_to_outfile(self, s):
        self.outfile.write(s)


def maybe_print_count(i, freq):
    if i - int(i / freq) * freq == 0:
        print(i)


def preprocess_one_line(sentence, lemmatizer, lemmatize_flag, stop_words):
    from nltk import word_tokenize
    from nltk.tag import pos_tag
    import re
    new_sentence = ''
    words = word_tokenize(sentence)
    for tok, tag in pos_tag(words):
        tok = tok.rstrip().lower()  # Convert to lower case
        tok = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                   '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', tok)  # Remove hyperlinks
        tok = re.sub("(@[A-Za-z0-9_]+)","", tok)  # Remove @ mentions
        tok = tok.replace("won't", "will not")
        tok = tok.replace("n't", " not")
        tok = tok.replace("'ll", " will")
        tok = tok.replace("'s", " is")
        tok = tok.replace("'ve", " have")
        tok = tok.replace("'re", " are")
        tok = tok.replace("'d", " would")
        tok = re.sub("[\"\'\`\#\@\.\,\!\?\:\;\-\=]", "", tok)  # Remove punctuation
        if lemmatize_flag:
            if tag.startswith('NN'):
                pos = 'n'
            elif tag.startswith('VB'):
                pos = 'v'
            else:
                pos = 'a'
            tok = lemmatizer.lemmatize(tok, pos)  # Lemmatize token
        if tok not in stop_words:
            new_sentence += tok
            new_sentence += ' '
    return new_sentence


def convert_label_to_value(label):
    if re.search('positive', label, re.IGNORECASE):
        return 1
    elif re.search('negative', label, re.IGNORECASE):
        return -1
    else:
        return 0

def remove_duplicates(words):
    return(list(dict.fromkeys(words)))


