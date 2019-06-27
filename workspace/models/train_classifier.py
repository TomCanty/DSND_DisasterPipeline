import sys
import pandas as pd
from sqlalchemy import create_engine
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report

import pickle

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """Class extention of sklearn fit and transform, to identify if the first work of a sentence is a verb or RT"""

    def starting_verb(self, text):
        """Parses the messages into sentences and words, identifies if the first word is verb or RT

        Arguments:
        text - (str) message

        Return:
            Boolean
        """
        sentence_list = nltk.sent_tokenize(re.sub('[^\w\s]',' ',text))
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            try: 
                first_word, first_tag = pos_tags[0]
            except:
                print(sentence_list,sentence,pos_tags)
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        """Transform message into boolean per the starting_verb function"""
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)
    pass


def load_data(database_filepath):
    """ Loads the sqlite database per the database_filepath 
    
    Loads the "Messages" Table froom SQLite database which was created by process_data.py.


    Arguments:
    database_filepath (str) - sqlite database filepath and name
    
    Returns:
    X (Dataframe) - a dataframe of all of the messages from the database
    Y (Dateframe) - a dataframe of how each meassage is classified over a couple dozen categories
    Y.columns (list) - a list of categories
    """
    dbpath = 'sqlite:///' + database_filepath
    engine = create_engine(dbpath)
    df = pd.read_sql('SELECT * FROM "Messages"', engine)
    X = df['message']
    Y = df.drop(columns=['message','original','genre'])

    return X,Y, list(Y.columns)


def tokenize(text):
    """ cleans and tokenizes text string and returns tokens 
    
    arguments:
    text (str) - message string
    
    """

    tokens = word_tokenize(re.sub('[^\w\s]',' ',text))
    lem = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lem.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
       
    return clean_tokens


def build_model():
    """ Builds the gridsearched ML pipeline  """

    pipeline = Pipeline([
        ('features',FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
            ('starting_verb', StartingVerbExtractor())
        ])),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ])
    
    parameters = {'clf__estimator__max_depth': [500],
    'clf__estimator__max_features': ['auto'],
    'clf__estimator__min_samples_leaf': [1],
    'clf__estimator__min_samples_split': [2]}

    cv = GridSearchCV(pipeline,parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluates a fitted model, prints classification report for each category

    Arguments:
    model - pipeline model built using the build_model funtion
    X_test - X data
    Y_test - 
    """
    Y_predict = model.predict(X_test)
    for i in range(Y_test.shape[1]):
        print(category_names[i])
        print(classification_report(Y_predict[:,i],Y_test.values[:,i]))
    pass


def save_model(model, model_filepath):
    """ saves the model,via pickle, to the model_filepath"""
    pickle.dump(model, open(model_filepath, 'wb'))
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        print('Xtrain: ', X_train.shape)
        print('Ytrain: ', Y_train.shape)
        print(model)
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()