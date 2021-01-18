import fasttext
from common_functions import convert_label_to_value, SentimentScores

def train_fastext(num_epochs):
    model = fasttext.train_supervised('/home/data/fastext_train_tweets.txt', epoch=num_epochs, verbose=2)
    test_file = open('/home/data/fastext_test_tweets.txt', 'r')
    scores = SentimentScores('fastext'+str(num_epochs))
    revs = test_file.readlines()
    for rev in revs:
        rev = rev.replace('\n', '')
        words = rev.split(' ')
        label = words[0]
        gt = convert_label_to_value(label)
        ln = len(rev)
        if label == '__label__NEUTRAL':
            s = rev[16:ln]
        else:
            s = rev[17:ln]
        pred = model.predict(s)
        computed_score = convert_label_to_value(pred[0][0])
        scores.update(computed_score, gt)

    outfile = scores.output_results()
    outfile.close()

train_fastext('tweets', 100)
