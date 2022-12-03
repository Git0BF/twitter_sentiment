# Import necessary libraries
import streamlit as st
import tweepy
import re
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# Load the trained classifier
classifier = joblib.load('sentiment_classifier.pkl')

# Create a function to clean and preprocess the text data
def clean_text(text):
    # Remove punctuation and special characters
    text = re.sub('[^a-zA-Z0-9\s]', '', text)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    text = [word for word in text.split() if not word in stop_words]

    # Return the preprocessed text
    return " ".join(text)

# Create a function to analyze the sentiment of tweets from a given account
def analyze_sentiment(username):
    # Use the tweepy.OAuthHandler class to create an OAuth handler
    consumer_key = "YOUR_CONSUMER_KEY"
    consumer_secret = "YOUR_CONSUMER_SECRET"
    access_token = "YOUR_ACCESS_TOKEN"
    access_token_secret = "YOUR_ACCESS_TOKEN_SECRET"
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    # Use the tweepy.API class to create an API object
    api = tweepy.API(auth)

    # Use the tweepy.Cursor class to access the tweets from the specified account
    tweets = tweepy.Cursor(api.search, q=username, lang="en").items(100)

    # Create an empty list to store the tweets
    tweets_list = []

    # Iterate over the tweets and add them to the list
    for tweet in tweets:
        tweets_list.append(tweet.text)

    # Preprocess the tweets
    tweets_list = [clean_text(tweet) for tweet in tweets_list]

    # Use the CountVectorizer to create a document-term matrix
    vectorizer = CountVectorizer()
    dtm = vectorizer.fit_transform(tweets_list)

    # Use the trained classifier to predict the sentiments of the tweets
    predictions = classifier.predict(dtm)

    # Return the predictions
    return predictions

# Use streamlit to create a simple user interface
st.title("Twitter Sentiment Analysis")

# Specify the list of accounts to analyze
accounts = ['@realdonaldtrump', '@joebiden', '@elonmusk', '@billgates']

# Create an empty dictionary to store the predictions for each account
predictions_dict = {}

# Iterate over the list of accounts and analyze the sentiment of their tweets
for account in accounts:
    predictions_dict[account] = analyze_sentiment(account)

# Use streamlit to display the results
st.header("Sentiment Analysis Results")

# Iterate over the dictionary of predictions and use streamlit to display the results for each account
for account, predictions in predictions_dict.items():
    st.header(account)
    st.bar_chart(predictions)

