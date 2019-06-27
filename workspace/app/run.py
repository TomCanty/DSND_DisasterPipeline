import json
import plotly
import pandas as pd

import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine

from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report


app = Flask(__name__)


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

def tokenize(text):
    """ cleans and tokenizes text string and returns tokens 
    
    arguments:
    text (str) - message string
    
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    ## categorical analysis of each message genra\e
    direct = df[df['genre']=='direct'].iloc[:,4:]
    social = df[df['genre']=='social'].iloc[:,4:]
    news = df[df['genre']=='news'].iloc[:,4:]

    direct_percent = direct.sum()/direct.shape[0]
    social_percent = social.sum()/social.shape[0]
    news_percent = news.sum()/news.shape[0]
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count",
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data' :[
                Bar(
                    x= list(direct_percent.index),
                    y= list(direct_percent.values)
                )
            ],

            'layout': {
                'title': 'Direct Message Classification',
                'yaxis': {
                    'title' : 'Percent of "Direct" messages',
                    'tickformat': '%',
                    'range' : [0,0.7],
                    'dtick' : 0.1
                },
                'xaxis' :{
                    'title' : 'Classificaiton ',
                    'tickangle': 45,
                    'tickfont' : {
                        'size': 10,
                        'color' : 'black'
                    }
                }
            }
        },
        {
            'data' :[
                Bar(
                    x=social_percent.index,
                    y=social_percent.values
                )
            ],

            'layout': {
                'title': 'Social Message Classification',
                'yaxis': {
                    'title' : 'Percent of "Social" messages',
                    'tickformat': '%',
                    'range' : [0,0.7],
                    'dtick' : 0.1
                },
                'xaxis' :{
                    'title' : 'Classificaiton ',
                    'tickangle': 45,
                    'tickfont' : {
                        'size': 10,
                        'color' : 'black'
                    }
                }
            }
        },
        {
            'data' :[
                Bar(
                    x= news_percent.index,
                    y=news_percent.values
                )
            ],

            'layout': {
                'title': 'News Message Classification',
                'yaxis': {
                    'title' : 'Percent of "News" messages',
                    'tickformat': '%',
                    'range' : [0,0.7],
                    'dtick' : 0.1
                },
                'xaxis' :{
                    'title' : 'Classificaiton ',
                    'tickangle': 45,
                    'tickfont' : {
                        'size': 10,
                        'color' : 'black'
                    }
                }
            }
        }

    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()