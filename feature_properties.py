import re
from collections import Counter
from nltk.util import ngrams
from nltk.sentiment import SentimentIntensityAnalyzer

class FeatureExtractor:
    def __init__(self):
        # Initialize sentiment analyzer
        self.sid = SentimentIntensityAnalyzer()
        
        # Define emoticon categories
        self.emoticons = {
            'positive': [':-)', ':)', '(:', '(-:', ':-D', ':D', ':-)', '😊', '🙂', '😀'],
            'negative': [':-(', ':(', '):', ')-:', ':-/', ':/', '😞', '😠', '😩'],
            'sarcastic': [';-]', ';^)', '>:[', ':-P', ';P', '😏', '😒', '🤨', '🙃']
        }
        
        # Punctuation patterns
        self.punctuation = ['!', '?', '!!', '??', '...', '?!', '!?']
        
    def get_ngram_features(self, text, n=3):
        chars = list(text)
        return {'char_ngram_' + ''.join(ng): chars.count(ng) 
            for ng in ngrams(chars, n) if len(ng) == n}

    def get_punctuation_features(self, text):
        """Count specific punctuation patterns"""
        return {
            'num_!': text.count('!'),
            'num_?': text.count('?'),
            'num_!!': text.count('!!'),
            'num_??': text.count('??'),
            'num_...': text.count('...'),
            'num_!?': text.count('!?'),
            'num_?!': text.count('?!')
        }

    def get_emoticon_features(self, text):
        """Count emoticons by category"""
        counts = Counter()
        for cat in self.emoticons:
            for emo in self.emoticons[cat]:
                counts[cat + '_emoticons'] += text.count(emo)
        return dict(counts)

    def get_sentiment_features(self, text):
        """Get sentiment scores using VADER"""
        scores = self.sid.polarity_scores(text)
        return {
            'sent_compound': scores['compound'],
            'sent_neg': scores['neg'],
            'sent_neu': scores['neu'],
            'sent_pos': scores['pos']
        }

    def get_ngram_features(self, text, n=3):
        """Character n-gram features"""
        chars = list(text)
        return {f'char_ngram_{"".join(ng)}': chars.count(ng) 
                for ng in ngrams(chars, n) if len("".join(ng)) == n}

    def get_all_features(self, text):
        features = {}
    # Ensure numeric conversions
        features.update({k: int(v) for k,v in self.get_punctuation_features(text).items()})
        features.update({k: int(v) for k,v in self.get_emoticon_features(text).items()})
        features.update({k: float(v) for k,v in self.get_sentiment_features(text).items()})
        features.update({k: int(v) for k,v in self.get_ngram_features(text).items()})
        return features

# Note: Run once before first use to download VADER lexicon
# import nltk
# nltk.download('vader_lexicon')