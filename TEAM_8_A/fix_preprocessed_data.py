#!/usr/bin/env python3
"""
Fix preprocessed_data.pkl by recreating vectorizers and scalers
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler
import re

def clean_text(text):
    """Clean and preprocess text"""
    if pd.isna(text) or text == '':
        return ''
    
    text = str(text).lower()
    # Remove URLs
    text = re.sub(r'http[s]?://\S+', '', text)
    # Remove emails
    text = re.sub(r'\S+@\S+', '', text)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove special characters but keep spaces
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_features(text):
    """Extract numerical features from text"""
    if pd.isna(text) or text == '':
        text = ''
    
    text = str(text)
    
    features = {
        'char_count': len(text),
        'word_count': len(text.split()),
        'avg_word_length': np.mean([len(word) for word in text.split()]) if text.split() else 0,
        'sentence_count': len(re.findall(r'[.!?]+', text)),
        'special_char_count': len(re.findall(r'[^a-zA-Z0-9\s]', text)),
        'digit_count': len(re.findall(r'\d', text)),
        'upper_case_count': len(re.findall(r'[A-Z]', text)),
        'lower_case_count': len(re.findall(r'[a-z]', text)),
        'upper_case_ratio': len(re.findall(r'[A-Z]', text)) / len(text) if len(text) > 0 else 0,
        'exclamation_count': text.count('!'),
        'question_count': text.count('?'),
        'url_count': len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)),
        'www_count': len(re.findall(r'www\.', text, re.IGNORECASE)),
        'short_url_count': len(re.findall(r'bit\.ly|tinyurl|goo\.gl', text, re.IGNORECASE)),
        'ip_address_count': len(re.findall(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', text)),
        'currency_symbol_count': len(re.findall(r'[$£€¥]', text)),
        'money_mention_count': len(re.findall(r'\$\d+|\d+\s*(?:dollar|usd|euro|pound)', text, re.IGNORECASE)),
        'spam_keyword_count': len(re.findall(r'\b(free|winner|click|buy|offer|prize|discount)\b', text, re.IGNORECASE)),
        'urgency_word_count': len(re.findall(r'\b(urgent|immediately|act now|limited time|expire)\b', text, re.IGNORECASE)),
        'suspicious_phrase_count': len(re.findall(r'congratulations|you have won|claim now|limited offer', text, re.IGNORECASE)),
        'email_address_count': len(re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)),
        'phone_number_count': len(re.findall(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text)),
        'html_tag_count': len(re.findall(r'<[^>]+>', text))
    }
    
    return list(features.values())

def main():
    print("="*60)
    print("FIXING PREPROCESSED DATA")
    print("="*60)
    
    # Load original data
    print("\n[1/5] Loading original preprocessed CSV...")
    try:
        df = pd.read_csv(r"C:\Users\sayen\anaconda3\envs\spam_detection\spam_project\spam_preprocessed.csv")
        print(f"✓ Loaded {len(df)} records")
    except FileNotFoundError:
        print("❌ spam_preprocessed.csv not found!")
        print("Please run step3_preprocessing.py first.")
        return
    
    # Check for required columns
    if 'cleaned_text' not in df.columns or 'Spam/Ham' not in df.columns:
        print("❌ Required columns not found!")
        print(f"Available columns: {df.columns.tolist()}")
        return
    
    # Split data (same as original)
    print("\n[2/5] Splitting data...")
    from sklearn.model_selection import train_test_split
    
    # Encode labels
    label_mapping = {'ham': 0, 'spam': 1}
    y = df['Spam/Ham'].map(label_mapping)
    
    # First split: train+val (80%) and test (20%)
    train_val_idx, test_idx = train_test_split(
        range(len(df)), test_size=0.2, random_state=42, stratify=y
    )
    
    # Second split: train (70% of total) and val (10% of total)
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=0.125, random_state=42, 
        stratify=y.iloc[train_val_idx]
    )
    
    print(f"✓ Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    # Get cleaned text
    train_texts = df.iloc[train_idx]['cleaned_text'].fillna('').tolist()
    val_texts = df.iloc[val_idx]['cleaned_text'].fillna('').tolist()
    test_texts = df.iloc[test_idx]['cleaned_text'].fillna('').tolist()
    
    y_train = y.iloc[train_idx].values
    y_val = y.iloc[val_idx].values
    y_test = y.iloc[test_idx].values
    
    # Create and fit TF-IDF vectorizer
    print("\n[3/5] Creating TF-IDF vectorizer...")
    tfidf_vectorizer = TfidfVectorizer(
        max_features=3000,
        min_df=2,
        max_df=0.8,
        ngram_range=(1, 2)
    )
    
    X_train_tfidf = tfidf_vectorizer.fit_transform(train_texts)
    X_val_tfidf = tfidf_vectorizer.transform(val_texts)
    X_test_tfidf = tfidf_vectorizer.transform(test_texts)
    
    print(f"✓ TF-IDF vocabulary size: {len(tfidf_vectorizer.vocabulary_)}")
    print(f"✓ Train shape: {X_train_tfidf.shape}")
    
    # Create Count Vectorizer
    print("\n[4/5] Creating Count vectorizer...")
    count_vectorizer = CountVectorizer(
        max_features=3000,
        min_df=2,
        max_df=0.8,
        ngram_range=(1, 2)
    )
    
    X_train_count = count_vectorizer.fit_transform(train_texts)
    X_val_count = count_vectorizer.transform(val_texts)
    X_test_count = count_vectorizer.transform(test_texts)
    
    print(f"✓ Count vocabulary size: {len(count_vectorizer.vocabulary_)}")
    
    # Extract and scale engineered features
    print("\n[5/5] Extracting and scaling features...")
    
    # Get feature columns
    feature_cols = [col for col in df.columns if col not in ['Message ID', 'Subject', 'Message', 'Spam/Ham', 'Date', 'cleaned_text']]
    
    if len(feature_cols) == 0:
        # Extract features from original message
        print("   Extracting features from messages...")
        if 'Message' in df.columns:
            features_list = df['Message'].fillna('').apply(lambda x: extract_features(x)).tolist()
            X_features = np.array(features_list)
        else:
            print("   ⚠️ No Message column found, using zero features")
            X_features = np.zeros((len(df), 23))
    else:
        X_features = df[feature_cols].fillna(0).values
    
    # Split features
    X_train_features = X_features[train_idx]
    X_val_features = X_features[val_idx]
    X_test_features = X_features[test_idx]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_features)
    X_val_scaled = scaler.transform(X_val_features)
    X_test_scaled = scaler.transform(X_test_features)
    
    print(f"✓ Feature shape: {X_train_scaled.shape}")
    
    # Create the data dictionary
    print("\n[6/6] Saving fixed preprocessed data...")
    
    preprocessed_data = {
        'X_train_tfidf': X_train_tfidf,
        'X_val_tfidf': X_val_tfidf,
        'X_test_tfidf': X_test_tfidf,
        'X_train_count': X_train_count,
        'X_val_count': X_val_count,
        'X_test_count': X_test_count,
        'X_train_scaled': X_train_scaled,
        'X_val_scaled': X_val_scaled,
        'X_test_scaled': X_test_scaled,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'tfidf_vectorizer': tfidf_vectorizer,
        'count_vectorizer': count_vectorizer,
        'scaler': scaler,
        'label_mapping': label_mapping,
        'feature_names': feature_cols if feature_cols else ['feature_' + str(i) for i in range(23)]
    }
    
    # Save
    with open('preprocessed_data.pkl', 'wb') as f:
        pickle.dump(preprocessed_data, f)
    
    print("✓ Saved to: preprocessed_data.pkl")
    
    # Verify the save
    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)
    
    print("\nTesting reload...")
    with open('preprocessed_data.pkl', 'rb') as f:
        test_data = pickle.load(f)
    
    print("✓ Data reloaded successfully!")
    print(f"✓ Keys: {list(test_data.keys())}")
    
    # Test vectorizer
    print("\nTesting TF-IDF vectorizer...")
    test_text = "This is a test email"
    try:
        test_transform = test_data['tfidf_vectorizer'].transform([test_text])
        print(f"✓ TF-IDF vectorizer works! Shape: {test_transform.shape}")
    except Exception as e:
        print(f"❌ TF-IDF vectorizer failed: {e}")
    
    # Test scaler
    print("\nTesting scaler...")
    try:
        test_features = np.random.rand(1, X_train_scaled.shape[1])
        test_scaled = test_data['scaler'].transform(test_features)
        print(f"✓ Scaler works! Shape: {test_scaled.shape}")
    except Exception as e:
        print(f"❌ Scaler failed: {e}")
    
    print("\n" + "="*60)
    print("✓ FIX COMPLETE!")
    print("="*60)
    print("\nYou can now run: streamlit run spam_detection_ui.py")

if __name__ == "__main__":
    main()