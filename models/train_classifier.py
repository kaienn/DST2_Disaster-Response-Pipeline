import sys
import re
import pandas as pd
import pickle
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
#nltk.download(['punkt', 'wordnet', 'stopwords'])
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report

def load_data(database_filepath):
    """
    load data from database (cleaned data)
    Inputs:
        database_filepath: path to database
    Outputs:
        X: numpy.ndarray. Disaster messages.
        Y: numpy.ndarray. Disaster categories for each messages.
    """
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql_table('CleanData', con=engine)
    X = df['message'].values
    Y = df.iloc[:,4:].values
    category_names = df.columns[4:]
    return X, Y, category_names


def tokenize(text):
    """Tokenize a text (a disaster message).
    Inputs:
        text, string. A disaster message.
    Outputs:
        array of tokenized, cleaned, and lemmatized text
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
        
    # normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # tokenize text
    tokens = word_tokenize(text)
    # Remove stopwords
    tokens = [t for t in tokens if t not in stopwords.words('english')]
    
    # initiate lemmatizer and lemmatize
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    build classification model 
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
    #'vect__max_df': (0.5, 0.75, 1.0),
    # 'vect__max_features': (None, 5000, 10000, 50000),
    'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    #'tfidf__use_idf': (True, False),
    # 'tfidf__norm': ('l1', 'l2'),
    #'clf__alpha': (0.00001, 0.000001),
    #'clf__estimator__n_estimators': [50, 100],
    #'clf__estimator__min_samples_split': [2, 4],
    # 'clf__max_iter': (10, 50, 80),
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1, verbose=10)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluate model's performance on test data
    Input: model, test data set
    Output: the Classification report on each category
    '''
    y_pred = model.predict(X_test)
    for i in range(0, len(category_names)):
        print(category_names[i])
        print(classification_report(Y_test[:, i], y_pred[:, i]))


def save_model(model, model_filepath):
    """
    export model as a pickel file
    """
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
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