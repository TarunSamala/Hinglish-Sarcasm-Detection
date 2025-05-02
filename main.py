import pandas as pd
from preprocessing import preprocess_data
from build_feature_vector import (
    build_feature_pipeline,
    extract_handcrafted_features,
    create_feature_matrix
)
from classification import train_classifier, evaluate_and_save
from sklearn.model_selection import train_test_split

def load_dataset():
    tweets = {}
    with open("Dataset/Sarcasm_tweets.txt", "r", encoding='utf-8') as f:
        current_id = ""
        for line in f:
            line = line.strip()
            if line.isdigit():
                current_id = line
            elif current_id:
                tweets[current_id] = line

    labels = {}
    with open("Dataset/Sarcasm_tweet_truth.txt", "r", encoding='utf-8') as f:
        current_id = ""
        for line in f:
            line = line.strip()
            if line.isdigit():
                current_id = line
            elif current_id:
                labels[current_id] = 1 if line == "YES" else 0

    valid_ids = [id for id in tweets if id in labels]
    return pd.DataFrame({
        'text': [tweets[id] for id in valid_ids],
        'label': [labels[id] for id in valid_ids]
    })

def main():
    df = load_dataset()
    df['clean_text'] = preprocess_data(df['text'])
    
    # Feature engineering
    feature_pipeline = build_feature_pipeline()
    tfidf_features = feature_pipeline.fit_transform(df['clean_text'])
    handcrafted_features = extract_handcrafted_features(df['clean_text'])
    
    # Create feature matrix
    X = create_feature_matrix(tfidf_features, handcrafted_features)
    y = df['label']
    
    # Data validation
    assert not X.isna().any().any(), "NaN values present in feature matrix"
    assert (X.dtypes == 'float32').all(), "Non-float32 columns present"
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Train and evaluate models
    print("Training SVM...")
    svm_model = train_classifier(X_train, y_train, 'svm')
    evaluate_and_save(svm_model, X_test, y_test, 'SVM')

    print("\nTraining Random Forest...")
    rf_model = train_classifier(X_train, y_train, 'rf')
    evaluate_and_save(rf_model, X_test, y_test, 'RandomForest')

if __name__ == "__main__":
    main()