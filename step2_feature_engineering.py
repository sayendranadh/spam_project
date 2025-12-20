"""
SPAM EMAIL DETECTION - ML CYBERSECURITY PROJECT
Step 2: Feature Engineering - Extract Cybersecurity Features
"""

import pandas as pd
import numpy as np
import re
import string
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# NLP imports
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')

class FeatureEngineer:
    """Extract cybersecurity-relevant features from email data"""
    
    def __init__(self, df, text_col='Message', label_col='Spam/Ham'):
        self.df = df.copy()
        self.text_col = text_col
        self.label_col = label_col
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        
    def extract_basic_features(self):
        """Extract basic text features"""
        print("\n[1/6] Extracting basic text features...")
        
        # Handle missing values
        self.df[self.text_col] = self.df[self.text_col].fillna('')
        
        # Character count
        self.df['char_count'] = self.df[self.text_col].apply(len)
        
        # Word count
        self.df['word_count'] = self.df[self.text_col].apply(lambda x: len(str(x).split()))
        
        # Average word length
        self.df['avg_word_length'] = self.df[self.text_col].apply(
            lambda x: np.mean([len(word) for word in str(x).split()]) if len(str(x).split()) > 0 else 0
        )
        
        # Sentence count
        self.df['sentence_count'] = self.df[self.text_col].apply(
            lambda x: len(re.findall(r'[.!?]+', str(x)))
        )
        
        print(f"   ✓ Created: char_count, word_count, avg_word_length, sentence_count")
        
    def extract_special_char_features(self):
        """Extract special character features (cybersecurity indicators)"""
        print("\n[2/6] Extracting special character features...")
        
        # Count of special characters
        self.df['special_char_count'] = self.df[self.text_col].apply(
            lambda x: len([c for c in str(x) if c in string.punctuation])
        )
        
        # Count of digits
        self.df['digit_count'] = self.df[self.text_col].apply(
            lambda x: len([c for c in str(x) if c.isdigit()])
        )
        
        # Count of uppercase letters
        self.df['upper_case_count'] = self.df[self.text_col].apply(
            lambda x: len([c for c in str(x) if c.isupper()])
        )
        
        # Count of lowercase letters
        self.df['lower_case_count'] = self.df[self.text_col].apply(
            lambda x: len([c for c in str(x) if c.islower()])
        )
        
        # Ratio of uppercase to total letters
        self.df['upper_case_ratio'] = self.df.apply(
            lambda row: row['upper_case_count'] / (row['upper_case_count'] + row['lower_case_count']) 
            if (row['upper_case_count'] + row['lower_case_count']) > 0 else 0, axis=1
        )
        
        # Exclamation marks (urgency indicator)
        self.df['exclamation_count'] = self.df[self.text_col].apply(
            lambda x: str(x).count('!')
        )
        
        # Question marks
        self.df['question_count'] = self.df[self.text_col].apply(
            lambda x: str(x).count('?')
        )
        
        print(f"   ✓ Created: special_char_count, digit_count, upper_case_count, etc.")
        
    def extract_url_features(self):
        """Extract URL-related features (phishing indicators)"""
        print("\n[3/6] Extracting URL features...")
        
        # URL patterns
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        www_pattern = r'www\.[\w\.-]+'
        
        # Count of URLs
        self.df['url_count'] = self.df[self.text_col].apply(
            lambda x: len(re.findall(url_pattern, str(x)))
        )
        
        # Count of www
        self.df['www_count'] = self.df[self.text_col].apply(
            lambda x: len(re.findall(www_pattern, str(x)))
        )
        
        # Count of shortened URLs (bit.ly, tinyurl, etc.)
        short_url_pattern = r'bit\.ly|tinyurl|goo\.gl|ow\.ly|t\.co'
        self.df['short_url_count'] = self.df[self.text_col].apply(
            lambda x: len(re.findall(short_url_pattern, str(x), re.IGNORECASE))
        )
        
        # IP addresses in URLs (suspicious)
        ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
        self.df['ip_address_count'] = self.df[self.text_col].apply(
            lambda x: len(re.findall(ip_pattern, str(x)))
        )
        
        print(f"   ✓ Created: url_count, www_count, short_url_count, ip_address_count")
        
    def extract_monetary_features(self):
        """Extract money/financial features (common in spam)"""
        print("\n[4/6] Extracting monetary features...")
        
        # Currency symbols
        self.df['currency_symbol_count'] = self.df[self.text_col].apply(
            lambda x: len(re.findall(r'[$€£¥₹]', str(x)))
        )
        
        # Money amounts
        money_pattern = r'\$\s?\d+(?:,\d{3})*(?:\.\d{2})?|\d+(?:,\d{3})*(?:\.\d{2})?\s?(?:dollars|USD|EUR|pounds)'
        self.df['money_mention_count'] = self.df[self.text_col].apply(
            lambda x: len(re.findall(money_pattern, str(x), re.IGNORECASE))
        )
        
        print(f"   ✓ Created: currency_symbol_count, money_mention_count")
        
    def extract_spam_keyword_features(self):
        """Extract spam keyword features"""
        print("\n[5/6] Extracting spam keyword features...")
        
        # Common spam keywords
        spam_keywords = [
            'free', 'win', 'winner', 'cash', 'prize', 'bonus', 'click', 'buy', 
            'order', 'urgent', 'limited', 'offer', 'discount', 'guarantee',
            'congratulations', 'claim', 'act now', 'call now', 'subscribe',
            'unsubscribe', 'viagra', 'pharmacy', 'loan', 'debt', 'credit'
        ]
        
        # Count spam keywords
        self.df['spam_keyword_count'] = self.df[self.text_col].apply(
            lambda x: sum([1 for keyword in spam_keywords if keyword in str(x).lower()])
        )
        
        # Urgency words
        urgency_words = ['urgent', 'immediate', 'act now', 'limited time', 'hurry', 'expires', 'today only']
        self.df['urgency_word_count'] = self.df[self.text_col].apply(
            lambda x: sum([1 for word in urgency_words if word in str(x).lower()])
        )
        
        # Suspicious phrases
        suspicious_phrases = ['click here', 'click below', 'confirm your account', 'verify your account', 
                             'update your information', 'suspended account', 'unusual activity']
        self.df['suspicious_phrase_count'] = self.df[self.text_col].apply(
            lambda x: sum([1 for phrase in suspicious_phrases if phrase in str(x).lower()])
        )
        
        print(f"   ✓ Created: spam_keyword_count, urgency_word_count, suspicious_phrase_count")
        
    def extract_email_specific_features(self):
        """Extract email-specific features"""
        print("\n[6/6] Extracting email-specific features...")
        
        # Email addresses count
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        self.df['email_address_count'] = self.df[self.text_col].apply(
            lambda x: len(re.findall(email_pattern, str(x)))
        )
        
        # Phone numbers
        phone_pattern = r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
        self.df['phone_number_count'] = self.df[self.text_col].apply(
            lambda x: len(re.findall(phone_pattern, str(x)))
        )
        
        # HTML tags (spam often contains HTML)
        self.df['html_tag_count'] = self.df[self.text_col].apply(
            lambda x: len(re.findall(r'<[^>]+>', str(x)))
        )
        
        print(f"   ✓ Created: email_address_count, phone_number_count, html_tag_count")
        
    def create_all_features(self):
        """Run all feature extraction methods"""
        print("\n" + "="*60)
        print("FEATURE ENGINEERING - EXTRACTING CYBERSECURITY FEATURES")
        print("="*60)
        
        self.extract_basic_features()
        self.extract_special_char_features()
        self.extract_url_features()
        self.extract_monetary_features()
        self.extract_spam_keyword_features()
        self.extract_email_specific_features()
        
        print("\n" + "="*60)
        print("✓ ALL FEATURES EXTRACTED SUCCESSFULLY!")
        print("="*60)
        
        # Show feature summary
        feature_cols = [col for col in self.df.columns if col not in 
                       ['Message ID', 'Subject', 'Message', 'Spam/Ham', 'Date']]
        
        print(f"\nTotal features created: {len(feature_cols)}")
        print(f"Feature names: {feature_cols}")
        
        return self.df
    
    def visualize_feature_importance(self):
        """Visualize feature distributions by class"""
        print("\nGenerating feature visualizations...")
        
        # Get feature columns
        feature_cols = [col for col in self.df.columns if col not in 
                       ['Message ID', 'Subject', 'Message', 'Spam/Ham', 'Date']]
        
        # Select top features to visualize
        top_features = feature_cols[:8]  # First 8 features
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.ravel()
        
        for idx, feature in enumerate(top_features):
            if self.label_col in self.df.columns:
                for label in self.df[self.label_col].unique():
                    data = self.df[self.df[self.label_col] == label][feature]
                    axes[idx].hist(data, alpha=0.6, label=label, bins=30)
                axes[idx].legend()
            else:
                axes[idx].hist(self.df[feature], bins=30)
            
            axes[idx].set_title(feature, fontweight='bold')
            axes[idx].set_xlabel('Value')
            axes[idx].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('feature_distributions.png', dpi=300, bbox_inches='tight')
        print("✓ Visualization saved as 'feature_distributions.png'")
        plt.show()
        
    def save_features(self, output_path='spam_features.csv'):
        """Save the featured dataset"""
        self.df.to_csv(output_path, index=False)
        print(f"\n✓ Featured dataset saved to: {output_path}")
        print(f"   Shape: {self.df.shape}")
        

# MAIN EXECUTION
if __name__ == "__main__":
    print("="*60)
    print("SPAM EMAIL DETECTION - STEP 2: FEATURE ENGINEERING")
    print("="*60)
    
    # Load the data
    data_path = r'C:\Users\sayen\anaconda3\envs\spam_detection\spam_project\enron_spam_data.csv'
    
    print("\nLoading data...")
    df = pd.read_csv(data_path, encoding='latin-1')
    print(f"✓ Data loaded! Shape: {df.shape}")
    
    # Initialize feature engineer
    engineer = FeatureEngineer(df, text_col='Message', label_col='Spam/Ham')
    
    # Extract all features
    df_featured = engineer.create_all_features()
    
    # Visualize features
    engineer.visualize_feature_importance()
    
    # Save featured dataset
    output_path = r"C:\Users\sayen\anaconda3\envs\spam_detection\spam_project\spam_features.csv"
    engineer.save_features(output_path)
    
    # Show sample of featured data
    print("\n" + "="*60)
    print("SAMPLE OF FEATURED DATA")
    print("="*60)
    feature_cols = [col for col in df_featured.columns if col not in 
                   ['Message ID', 'Subject', 'Message', 'Date']]
    print(df_featured[feature_cols].head())
    
    print("\n" + "="*60)
    print("✓ STEP 2 COMPLETE!")
    print("="*60)
    print("\nNext step: Data Preprocessing & Text Cleaning (Step 3)")