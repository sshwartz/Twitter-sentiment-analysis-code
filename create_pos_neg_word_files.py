from nltk.stem.wordnet import WordNetLemmatizer
from SentimentAnalysis.common_functions import preprocess_one_line, remove_duplicates


lemmatizer = WordNetLemmatizer()
lemmatize_flag = True

pos_raw_file = open('/home/data/positive-words-raw.txt', 'r')
neg_raw_file = open('/home/data/negative-words-raw.txt', 'r')
pos_file = open('/home/data/positive-words.txt', 'w')
neg_file = open('/home/data/negative-words.txt', 'w')
positives_raw = pos_raw_file.readlines()
negatives_raw = neg_raw_file.readlines()

positives = []
for pos_line in positives_raw:
    new_pos_line = preprocess_one_line(pos_line, lemmatizer, lemmatize_flag, []) # last arg is stopwords -- there are none by definition
    positives.append(new_pos_line)
positives = remove_duplicates(positives)

negatives = []
for neg_line in negatives_raw:
    new_neg_line = preprocess_one_line(neg_line, lemmatizer, lemmatize_flag, [])
    negatives.append(new_neg_line)
negatives = remove_duplicates(negatives)

for word in negatives:
    neg_file.write(word+'\n')
neg_file.close()

for word in positives:
    pos_file.write(word+'\n')
pos_file.close()
