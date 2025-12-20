"""
SPAM EMAIL DETECTION - ML CYBERSECURITY PROJECT
Step 3: Data Preprocessing & Text Cleaning
"""

import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# NLP imports
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

class TextPreprocessor:
    """Clean and preprocess text data for ML"""
    
    def __init__(self, df, text_col='Message', label_col='Spam/Ham'):
        self.df = df.copy()
        self.text_col = text_col
        self.label_col = label_col
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        
    def clean_text(self, text):
        """Clean individual text"""
        if pd.isna(text):
            return ""
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\.\S+', ' url ', text)
        
        # Remove email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', ' email ', text)
        
        # Remove phone numbers
        text = re.sub(r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', ' phone ', text)
        
        # Remove money amounts but keep indicator
        text = re.sub(r'\$\s?\d+(?:,\d{3})*(?:\.\d{2})?', ' money ', text)
        
        # Remove IP addresses
        text = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', ' ipaddress ', text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def remove_stopwords(self, text):
        """Remove stopwords from text"""
        words = text.split()
        filtered_words = [word for word in words if word not in self.stop_words]
        return ' '.join(filtered_words)
    
    def stem_text(self, text):
        """Apply stemming to text"""
        words = text.split()
        stemmed_words = [self.stemmer.stem(word) for word in words]
        return ' '.join(stemmed_words)
    
    def lemmatize_text(self, text):
        """Apply lemmatization to text"""
        words = text.split()
        lemmatized_words = [self.lemmatizer.lemmatize(word) for word in words]
        return ' '.join(lemmatized_words)
    
    def remove_punctuation(self, text):
        """Remove punctuation from text"""
        return text.translate(str.maketrans('', '', string.punctuation))
    
    def preprocess_all_texts(self, remove_stops=True, use_stemming=False, use_lemmatization=True):
        """Preprocess all texts in the dataset"""
        print("\n" + "="*60)
        print("TEXT PREPROCESSING PIPELINE")
        print("="*60)
        
        # Handle missing values
        print("\n[1/6] Handling missing values...")
        self.df[self.text_col] = self.df[self.text_col].fillna('')
        missing_before = self.df[self.text_col].isna().sum()
        print(f"   ✓ Missing values handled: {missing_before}")
        
        # Clean text
        print("\n[2/6] Cleaning text (URLs, emails, HTML, etc.)...")
        self.df['cleaned_text'] = self.df[self.text_col].apply(self.clean_text)
        print(f"   ✓ Text cleaned for {len(self.df)} messages")
        
        # Remove punctuation
        print("\n[3/6] Removing punctuation...")
        self.df['cleaned_text'] = self.df['cleaned_text'].apply(self.remove_punctuation)
        print(f"   ✓ Punctuation removed")
        
        # Remove stopwords
        if remove_stops:
            print("\n[4/6] Removing stopwords...")
            self.df['cleaned_text'] = self.df['cleaned_text'].apply(self.remove_stopwords)
            print(f"   ✓ Stopwords removed (kept {len(self.stop_words)} stopwords list)")
        else:
            print("\n[4/6] Skipping stopword removal...")
        
        # Stemming or Lemmatization
        if use_stemming:
            print("\n[5/6] Applying stemming...")
            self.df['cleaned_text'] = self.df['cleaned_text'].apply(self.stem_text)
            print(f"   ✓ Stemming applied")
        elif use_lemmatization:
            print("\n[5/6] Applying lemmatization...")
            self.df['cleaned_text'] = self.df['cleaned_text'].apply(self.lemmatize_text)
            print(f"   ✓ Lemmatization applied")
        else:
            print("\n[5/6] Skipping stemming/lemmatization...")
        
        # Final cleaning
        print("\n[6/6] Final text cleaning...")
        self.df['cleaned_text'] = self.df['cleaned_text'].apply(
            lambda x: re.sub(r'\s+', ' ', x).strip()
        )
        print(f"   ✓ Final cleaning complete")
        
        print("\n" + "="*60)
        print("✓ TEXT PREPROCESSING COMPLETE!")
        print("="*60)
        
    def show_examples(self, n=5):
        """Show before/after examples"""
        print("\n" + "="*60)
        print("BEFORE vs AFTER EXAMPLES")
        print("="*60)
        
        for i in range(min(n, len(self.df))):
            if len(str(self.df.iloc[i][self.text_col])) > 0:
                print(f"\n--- Example {i+1} ---")
                print(f"Label: {self.df.iloc[i][self.label_col]}")
                print(f"\nBEFORE: {str(self.df.iloc[i][self.text_col])[:200]}...")
                print(f"\nAFTER:  {self.df.iloc[i]['cleaned_text'][:200]}...")
                print("-" * 60)


class DataSplitter:
    """Split data into train/validation/test sets"""
    
    def __init__(self, df, text_col='cleaned_text', label_col='Spam/Ham'):
        self.df = df.copy()
        self.text_col = text_col
        self.label_col = label_col
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
    def encode_labels(self):
        """Encode labels to numeric values"""
        print("\n[1/3] Encoding labels...")
        self.df['label_encoded'] = self.label_encoder.fit_transform(self.df[self.label_col])
        
        # Show mapping
        label_mapping = dict(zip(self.label_encoder.classes_, 
                                self.label_encoder.transform(self.label_encoder.classes_)))
        print(f"   ✓ Label mapping: {label_mapping}")
        
        return label_mapping
    
    def split_data(self, test_size=0.2, val_size=0.1, random_state=42):
        """Split data into train, validation, and test sets"""
        print("\n[2/3] Splitting data...")
        
        # Get features and labels
        X = self.df[self.text_col]
        y = self.df['label_encoded']
        
        # Get numerical features
        feature_cols = [col for col in self.df.columns if col not in 
                       ['Message ID', 'Subject', 'Message', 'Spam/Ham', 'Date', 
                        'cleaned_text', 'label_encoded']]
        X_features = self.df[feature_cols]
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        X_features_temp, X_features_test = train_test_split(
            X_features, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            random_state=random_state, stratify=y_temp
        )
        X_features_train, X_features_val = train_test_split(
            X_features_temp, test_size=val_size_adjusted, 
            random_state=random_state, stratify=y_temp
        )
        
        print(f"   ✓ Train set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
        print(f"   ✓ Validation set: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
        print(f"   ✓ Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
        
        # Store splits
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        
        self.X_features_train = X_features_train
        self.X_features_val = X_features_val
        self.X_features_test = X_features_test
        
        return (X_train, X_val, X_test, y_train, y_val, y_test,
                X_features_train, X_features_val, X_features_test)
    
    def scale_features(self):
        """Scale numerical features"""
        print("\n[3/3] Scaling numerical features...")
        
        # Fit on training data only
        self.X_features_train_scaled = self.scaler.fit_transform(self.X_features_train)
        self.X_features_val_scaled = self.scaler.transform(self.X_features_val)
        self.X_features_test_scaled = self.scaler.transform(self.X_features_test)
        
        print(f"   ✓ Features scaled using StandardScaler")
        print(f"   ✓ Feature shape: {self.X_features_train_scaled.shape[1]} features")
        
    def visualize_splits(self):
        """Visualize data splits"""
        print("\nGenerating split visualizations...")
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Count plots for each split
        splits = [
            ('Train', self.y_train),
            ('Validation', self.y_val),
            ('Test', self.y_test)
        ]
        
        for idx, (name, y_data) in enumerate(splits):
            counts = pd.Series(y_data).value_counts()
            labels = [self.label_encoder.inverse_transform([i])[0] for i in counts.index]
            
            axes[idx].bar(labels, counts.values, color=['green', 'red'])
            axes[idx].set_title(f'{name} Set Distribution', fontweight='bold')
            axes[idx].set_xlabel('Class')
            axes[idx].set_ylabel('Count')
            axes[idx].tick_params(rotation=0)
            
            # Add percentages
            for i, (label, count) in enumerate(zip(labels, counts.values)):
                percentage = count / counts.sum() * 100
                axes[idx].text(i, count, f'{percentage:.1f}%', 
                             ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('data_splits.png', dpi=300, bbox_inches='tight')
        print("✓ Visualization saved as 'data_splits.png'")
        plt.show()


class TextVectorizer:
    """Convert text to numerical vectors"""
    
    def __init__(self):
        self.tfidf_vectorizer = None
        self.count_vectorizer = None
        
    def create_tfidf_features(self, X_train, X_val, X_test, max_features=5000):
        """Create TF-IDF features"""
        print("\n" + "="*60)
        print("CREATING TF-IDF FEATURES")
        print("="*60)
        
        print(f"\n[1/2] Fitting TF-IDF vectorizer on training data...")
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),  # unigrams and bigrams
            min_df=2,
            max_df=0.95
        )
        
        X_train_tfidf = self.tfidf_vectorizer.fit_transform(X_train)
        print(f"   ✓ TF-IDF vocabulary size: {len(self.tfidf_vectorizer.vocabulary_)}")
        
        print(f"\n[2/2] Transforming validation and test data...")
        X_val_tfidf = self.tfidf_vectorizer.transform(X_val)
        X_test_tfidf = self.tfidf_vectorizer.transform(X_test)
        
        print(f"   ✓ Train TF-IDF shape: {X_train_tfidf.shape}")
        print(f"   ✓ Val TF-IDF shape: {X_val_tfidf.shape}")
        print(f"   ✓ Test TF-IDF shape: {X_test_tfidf.shape}")
        
        # Show top features
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        print(f"\n   Top 20 features: {list(feature_names[:20])}")
        
        return X_train_tfidf, X_val_tfidf, X_test_tfidf
    
    def create_count_features(self, X_train, X_val, X_test, max_features=5000):
        """Create Count Vectorizer features"""
        print("\n" + "="*60)
        print("CREATING COUNT VECTORIZER FEATURES")
        print("="*60)
        
        print(f"\n[1/2] Fitting Count Vectorizer on training data...")
        self.count_vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
        X_train_count = self.count_vectorizer.fit_transform(X_train)
        print(f"   ✓ Count vocabulary size: {len(self.count_vectorizer.vocabulary_)}")
        
        print(f"\n[2/2] Transforming validation and test data...")
        X_val_count = self.count_vectorizer.transform(X_val)
        X_test_count = self.count_vectorizer.transform(X_test)
        
        print(f"   ✓ Train Count shape: {X_train_count.shape}")
        print(f"   ✓ Val Count shape: {X_val_count.shape}")
        print(f"   ✓ Test Count shape: {X_test_count.shape}")
        
        return X_train_count, X_val_count, X_test_count


# MAIN EXECUTION
if __name__ == "__main__":
    print("="*60)
    print("SPAM EMAIL DETECTION - STEP 3: DATA PREPROCESSING")
    print("="*60)
    
    # Load featured data
    data_path = r'C:\Users\sayen\anaconda3\envs\spam_detection\spam_project\spam_features.csv'
    
    print("\nLoading featured data...")
    df = pd.read_csv(data_path, encoding='latin-1')
    print(f"✓ Data loaded! Shape: {df.shape}")
    
    # === PART 1: TEXT PREPROCESSING ===
    print("\n" + "="*60)
    print("PART 1: TEXT PREPROCESSING")
    print("="*60)
    
    preprocessor = TextPreprocessor(df, text_col='Message', label_col='Spam/Ham')
    
    # Preprocess texts
    preprocessor.preprocess_all_texts(
        remove_stops=True,
        use_stemming=False,
        use_lemmatization=True
    )
    
    # Show examples
    preprocessor.show_examples(n=3)
    
    # Get preprocessed dataframe
    df_processed = preprocessor.df
    
    # === PART 2: DATA SPLITTING ===
    print("\n" + "="*60)
    print("PART 2: DATA SPLITTING")
    print("="*60)
    
    splitter = DataSplitter(df_processed, text_col='cleaned_text', label_col='Spam/Ham')
    
    # Encode labels
    label_mapping = splitter.encode_labels()
    
    # Split data
    (X_train, X_val, X_test, y_train, y_val, y_test,
     X_features_train, X_features_val, X_features_test) = splitter.split_data(
        test_size=0.2,
        val_size=0.1,
        random_state=42
    )
    
    # Scale features
    splitter.scale_features()
    
    # Visualize splits
    splitter.visualize_splits()
    
    # === PART 3: TEXT VECTORIZATION ===
    print("\n" + "="*60)
    print("PART 3: TEXT VECTORIZATION")
    print("="*60)
    
    vectorizer = TextVectorizer()
    
    # Create TF-IDF features
    X_train_tfidf, X_val_tfidf, X_test_tfidf = vectorizer.create_tfidf_features(
        X_train, X_val, X_test, max_features=3000
    )
    
    # Optionally create Count features
    X_train_count, X_val_count, X_test_count = vectorizer.create_count_features(
        X_train, X_val, X_test, max_features=3000
    )
    
    # === SAVE PROCESSED DATA ===
    print("\n" + "="*60)
    print("SAVING PROCESSED DATA")
    print("="*60)
    
    # Save preprocessed dataframe
    output_path = r'C:\Users\sayen\anaconda3\envs\spam_detection\spam_project\spam_preprocessed.csv'
    df_processed.to_csv(output_path, index=False)
    print(f"\n✓ Preprocessed data saved to: {output_path}")
    
    # Save numpy arrays for later use
    import pickle
    
    data_dict = {
        'X_train': X_train.values,
        'X_val': X_val.values,
        'X_test': X_test.values,
        'y_train': y_train.values,
        'y_val': y_val.values,
        'y_test': y_test.values,
        'X_train_tfidf': X_train_tfidf,
        'X_val_tfidf': X_val_tfidf,
        'X_test_tfidf': X_test_tfidf,
        'X_features_train_scaled': splitter.X_features_train_scaled,
        'X_features_val_scaled': splitter.X_features_val_scaled,
        'X_features_test_scaled': splitter.X_features_test_scaled,
        'label_encoder': splitter.label_encoder,
        'scaler': splitter.scaler,
        'tfidf_vectorizer': vectorizer.tfidf_vectorizer,
        'feature_names': list(X_features_train.columns)
    }
    
    pickle_path = r'C:\Users\sayen\anaconda3\envs\spam_detection\spam_project\preprocessed_data.pkl'
    with open(pickle_path, 'wb') as f:
        pickle.dump(data_dict, f)
    print(f"✓ Processed data saved to: {pickle_path}")
    
    # === SUMMARY ===
    print("\n" + "="*60)
    print("✓ STEP 3 COMPLETE!")
    print("="*60)
    
    print("\n📊 Data Summary:")
    print(f"   Total samples: {len(df_processed)}")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Validation samples: {len(X_val)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"\n📝 Text Features:")
    print(f"   TF-IDF features: {X_train_tfidf.shape[1]}")
    print(f"   Engineered features: {splitter.X_features_train_scaled.shape[1]}")
    print(f"\n🎯 Label Distribution:")
    print(f"   Classes: {list(label_mapping.keys())}")
    print(f"   Encoding: {label_mapping}")
    
    print("\n✨ Next step: Model Training (Step 4)")
    print("   - Logistic Regression")
    print("   - Random Forest")
    print("   - Naive Bayes")
    print("   - SVM")
    print("   - Deep Learning models")