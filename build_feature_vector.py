from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from feature_properties import FeatureExtractor
import pandas as pd

# Main feature pipeline builder
def build_feature_pipeline():
    return FeatureUnion([
        ('tfidf_char', TfidfVectorizer(
            analyzer='char',
            ngram_range=(3, 5),
            max_features=2000,
            lowercase=False
        )),
        ('tfidf_word', TfidfVectorizer(
            analyzer='word',
            token_pattern=r'\b\w+\b',
            ngram_range=(1, 2),
            max_features=3000,
            lowercase=False
        ))
    ])

# Handcrafted feature extraction
def extract_handcrafted_features(texts):
    extractor = FeatureExtractor()
    features = []
    
    # Initialize with first text's features to get all keys
    sample_features = extractor.get_all_features(texts[0])
    all_keys = sample_features.keys()
    
    for text in texts:
        feat = extractor.get_all_features(text)
        # Fill missing keys with 0
        for key in all_keys:
            if key not in feat:
                feat[key] = 0
        features.append(feat)
    
    return pd.DataFrame(features).fillna(0).astype(float)

def get_sarcasm_features(self, text):
    return {
        'irony_keywords': sum(1 for word in text.split() if word in {'yeah', 'right', 'sure'}),
        'quote_count': text.count('"') + text.count("'"),
        'caps_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0
    }

# Feature matrix creation
def create_feature_matrix(tfidf_features, handcrafted_features):
    # Convert TF-IDF features
    tfidf_df = pd.DataFrame.sparse.from_spmatrix(tfidf_features)
    tfidf_df.columns = [f'tfidf_{i}' for i in range(tfidf_df.shape[1])]
    
    # Combine features
    combined_df = pd.concat([tfidf_df, handcrafted_features], axis=1)
    combined_df.columns = combined_df.columns.astype(str)
    
    # Final cleanup
    return combined_df.fillna(0).astype('float32')