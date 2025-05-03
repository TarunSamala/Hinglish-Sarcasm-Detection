from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

def train_classifier(X, y, model_type='svm'):
    smote = SMOTE(random_state=42, sampling_strategy=0.5)
    
    if model_type == 'svm':
        pipeline = Pipeline([
            ('smote', smote),
            ('scaler', StandardScaler()),
            ('clf', SVC(
                kernel='rbf',
                class_weight={0:1, 1:15},
                probability=True,
                random_state=42
            ))
        ])
        params = {'clf__C': [0.1, 1, 10], 'clf__gamma': ['scale', 'auto']}
    
    elif model_type == 'rf':
        pipeline = Pipeline([
            ('smote', smote),
            ('clf', RandomForestClassifier(
                n_estimators=1000,
                class_weight={0:1, 1:12},
                max_depth=30,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            ))
        ])
        params = {'clf__max_features': ['sqrt', 'log2']}
        
    elif model_type == 'lr':
        pipeline = Pipeline([
            ('smote', smote),
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(
                class_weight={0:1, 1:15},
                solver='saga',
                penalty='l2',
                max_iter=1000,
                random_state=42
            ))
        ])
        params = {
            'clf__C': [0.01, 0.1, 1, 10],
            'clf__penalty': ['l1', 'l2']
        }
    
    # Threshold tuning
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    grid = GridSearchCV(pipeline, params, cv=3, scoring='f1_macro', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    best_model = grid.best_estimator_
    y_proba = best_model.predict_proba(X_val)[:, 1]
    
    # Find optimal threshold
    thresholds = np.linspace(0.1, 0.9, 50)
    best_threshold = 0.5
    best_f1 = 0
    
    for thresh in thresholds:
        y_pred = (y_proba > thresh).astype(int)
        current_f1 = f1_score(y_val, y_pred, zero_division=0)
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = thresh
    
    setattr(best_model, 'best_threshold', best_threshold)
    joblib.dump(best_model, f'{model_type}_model.pkl')
    
    return best_model

def evaluate_and_save(model, X_test, y_test, model_name):
    if hasattr(model, 'best_threshold'):
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba > model.best_threshold).astype(int)
    else:
        y_pred = model.predict(X_test)
    
    # Generate reports
    report = classification_report(
        y_test, y_pred,
        target_names=['Non-Sarcastic', 'Sarcastic'],
        output_dict=True,
        zero_division=0
    )
    
    # Save and print results
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(f'{model_name}_report.csv', float_format='%.2f')
    
    print(f"\n{model_name} Classification Report:")
    print(pd.DataFrame(report).transpose().to_string(float_format='%.2f'))
    
