from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout
from keras.models import Sequential
from xgboost import XGBClassifier
import pandas as pd
from imblearn.over_sampling import SMOTE
from common_functions import SentimentScores, convert_label_to_value

class SentimentClassifier(object):
    def __init__(self):
        self.X_train_BOW = None
        self.y_train = None
        self.X_test_BOW = None
        self.y_test = None
        self.num_features = 2000  # number of BOW entries
        self.test_pct = .3      # % of records to be used to test vs train
        self.features = None

    def run_classifier(self, clf):
        scores = SentimentScores(str(clf))
        clf.fit(self.X_train_BOW, self.y_train)
        n = self.y_test.shape[0] - 1
        for i in range(0, n):
            val = clf.predict(self.X_test_BOW[i:i + 1])[0]
            gt = self.y_test[i]
            scores.update(val, gt)
        scores.output_results()

        if 'DecisionTree' in str(clf):
            deciding_features = clf.tree_.feature
            scores.write_string_to_outfile('Top Decision Tree Features:\n')
            for i in range(0, 10):
                scores.write_string_to_outfile(self.features[deciding_features[i]] + "\n")

    def keras_classifier(self, epochs):
        nn_model = Sequential()  # The Sequential model is the simpler of the two types of Keras models
        nn_model.add(Dense(self.num_features, input_dim=self.num_features, activation='relu'))  # Add a hidden layer with ReLU activation
        nn_model.add(Dropout(.15))
        nn_model.add(Dense(1))  # Add an output layer
        nn_model.compile(loss="mse", optimizer="adam")  # Compile the model using mean-squared error as the error function and stochastic gradient descent
        nn_model.fit(self.X_train_BOW, self.y_train, verbose=0, epochs=epochs)
        predict = nn_model.predict(self.X_test_BOW)
        scores = SentimentScores('keras')
        n = self.y_test.shape[0] - 1
        for i in range(0, n):
            pred = predict[i][0]
            if pred < -.3333:
                val = -1
            elif pred > .3333:
                val = 1
            else:
                val = 0
            gt = self.y_test[i]
            scores.update(val, gt)
        scores.output_results()

    def classify(self, lassifier_type):
        df_all = pd.read_csv('/home/data/tweets_preprocessed.txt', header=0, names=['sentiment', 'review'])
        sentiment = df_all['sentiment']
        df_all['sentiment'] = sentiment.map(convert_label_to_value)
        X = df_all['review'].values
        y = df_all['sentiment'].values
        X_train, X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=.3, stratify=y)

        cv = CountVectorizer(ngram_range=(1,3), max_features=self.num_features)
        cv.fit(X_train)
        self.X_train_BOW = cv.transform(X_train)
        self.X_test_BOW = cv.transform(X_test)
        self.features = cv.get_feature_names()

        # Upsample for equal class sizes in training data
        sm = SMOTE()
        self.X_train_BOW, self.y_train = sm.fit_resample(self.X_train_BOW, self.y_train)

        if classifier_type == 'naive_bayes':
            self.run_classifier(MultinomialNB())
        elif classifier_type == 'decision_tree':
            clf = DecisionTreeClassifier()
            self.run_classifier(clf)
        elif classifier_type == 'xgboost':
            self.run_classifier(XGBClassifier())
        elif classifier_type == 'svc':
            self.run_classifier(LinearSVC())
        elif classifier_type == 'nearest_neighbor':
            self.run_classifier(KNeighborsClassifier())
        elif classifier_type == 'mlp':
            self.run_classifier(MLPClassifier())
        elif classifier_type == 'neural_network':
            for epochs in (5, 10, 20, 50):
                self.keras_classifier(epochs)
        elif classifier_type == 'all':
            for clf in (MultinomialNB(), DecisionTreeClassifier(random_state=0), XGBClassifier(), LinearSVC(random_state=0), KNeighborsClassifier()):
                self.run_classifier(clf)
            for epochs in (5, 10, 20, 50):
                self.keras_classifier(epochs)

sc = SentimentClassifier()
sc.classify('summary', 'all')
sc.classify('tweets', 'all')
sc.classify('detail', 'all')




