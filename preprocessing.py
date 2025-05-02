import re
import numpy as np
import emoji
from nltk.stem import SnowballStemmer

def clean_tweet(tweet):
    """Clean and normalize code-mixed tweets"""
    # Remove URLs
    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet)
    # Remove mentions
    tweet = re.sub(r'@\w+', '', tweet)
    # Remove special characters except code-mixed indicators
    tweet = re.sub(r'[^\w\s#!?।॰♥️★☆♡Ⓜ️〽️�]', '', tweet)
    # Normalize repeated characters
    tweet = re.sub(r'(.)\1{2,}', r'\1', tweet)
    # Convert to lowercase
    tweet = tweet.lower()
    return tweet.strip()

def preprocess_data(texts):
    """Main preprocessing pipeline"""
    processed = []
    for text in texts:
        text = clean_tweet(text)
        processed.append(text)
    return np.array(processed)