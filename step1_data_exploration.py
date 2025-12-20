"""
SPAM EMAIL DETECTION - ML CYBERSECURITY PROJECT
Step 1: Data Loading and Initial Exploration
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style for better visualizations
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

class SpamDataLoader:
    """Class to handle data loading and initial exploration"""
    
    def __init__(self, data_path):
        self.data_path = r'C:\Users\sayen\anaconda3\envs\spam_detection\spam_project\enron_spam_data.csv'
        self.df = None
        
    def load_data(self):
        """Load the spam dataset"""
        try:
            # For CSV files (most common)
            self.df = pd.read_csv(self.data_path, encoding='latin-1')
            print(f"✓ Data loaded successfully!")
            print(f"Shape: {self.df.shape}")
            return self.df
        except Exception as e:
            print(f"✗ Error loading data: {e}")
            print("\nTrying alternative encodings...")
            try:
                self.df = pd.read_csv(self.data_path, encoding='utf-8')
                print("✓ Loaded with UTF-8 encoding")
                return self.df
            except:
                print("✗ Failed to load data. Please check the file path and format.")
                return None
    
    def initial_exploration(self):
        """Perform initial data exploration"""
        if self.df is None:
            print("No data loaded. Please load data first.")
            return
        
        print("\n" + "="*50)
        print("INITIAL DATA EXPLORATION")
        print("="*50)
        
        # Basic info
        print("\n1. Dataset Info:")
        print(f"   Rows: {self.df.shape[0]}")
        print(f"   Columns: {self.df.shape[1]}")
        
        # Column names
        print("\n2. Column Names:")
        print(f"   {list(self.df.columns)}")
        
        # First few rows
        print("\n3. First 5 Rows:")
        print(self.df.head())
        
        # Data types
        print("\n4. Data Types:")
        print(self.df.dtypes)
        
        # Missing values
        print("\n5. Missing Values:")
        missing = self.df.isnull().sum()
        print(missing[missing > 0] if missing.sum() > 0 else "   No missing values!")
        
        # Basic statistics
        print("\n6. Basic Statistics:")
        print(self.df.describe())
        
    def analyze_labels(self, label_column='Spam/Ham'):
        """Analyze the distribution of spam vs ham"""
        if self.df is None:
            print("No data loaded.")
            return
        
        print("\n" + "="*50)
        print("LABEL DISTRIBUTION ANALYSIS")
        print("="*50)
        
        # Try to find the label column
        possible_names = ['Spam/Ham', 'label', 'class', 'Category', 'v1', 'target', 'spam']
        
        label_col = None
        for name in possible_names:
            if name in self.df.columns:
                label_col = name
                break
        
        if label_col is None:
            print(f"Label column not found. Available columns: {list(self.df.columns)}")
            return
        
        print(f"\nLabel column found: '{label_col}'")
        
        # Count distribution
        counts = self.df[label_col].value_counts()
        print(f"\nClass Distribution:")
        print(counts)
        
        # Percentages
        percentages = self.df[label_col].value_counts(normalize=True) * 100
        print(f"\nClass Percentages:")
        print(percentages)
        
        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Count plot
        counts.plot(kind='bar', ax=axes[0], color=['green', 'red'])
        axes[0].set_title('Spam vs Ham Count', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Class')
        axes[0].set_ylabel('Count')
        axes[0].tick_params(rotation=0)
        
        # Pie chart
        colors = ['#2ecc71', '#e74c3c']
        axes[1].pie(counts, labels=counts.index, autopct='%1.1f%%', 
                   colors=colors, startangle=90)
        axes[1].set_title('Class Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
        print("\n✓ Visualization saved as 'class_distribution.png'")
        plt.show()
        
        return label_col
    
    def text_length_analysis(self, text_column='text', label_column='label'):
        """Analyze text length characteristics"""
        if self.df is None:
            print("No data loaded.")
            return
        
        # Find text column
        possible_text_names = ['text', 'message', 'email', 'v2', 'Email Text', 'body']
        text_col = None
        for name in possible_text_names:
            if name in self.df.columns:
                text_col = name
                break
        
        if text_col is None:
            print(f"Text column not found. Available columns: {list(self.df.columns)}")
            return
        
        print(f"\n✓ Text column found: '{text_col}'")
        
        # Calculate text lengths
        self.df['text_length'] = self.df[text_col].astype(str).apply(len)
        self.df['word_count'] = self.df[text_col].astype(str).apply(lambda x: len(x.split()))
        
        print("\n" + "="*50)
        print("TEXT LENGTH ANALYSIS")
        print("="*50)
        
        print(f"\nOverall Statistics:")
        print(f"  Average character length: {self.df['text_length'].mean():.2f}")
        print(f"  Average word count: {self.df['word_count'].mean():.2f}")
        print(f"  Max character length: {self.df['text_length'].max()}")
        print(f"  Min character length: {self.df['text_length'].min()}")
        
        # By class
        if label_column in self.df.columns:
            print(f"\nBy Class:")
            for label in self.df[label_column].unique():
                subset = self.df[self.df[label_column] == label]
                print(f"\n  {label}:")
                print(f"    Avg character length: {subset['text_length'].mean():.2f}")
                print(f"    Avg word count: {subset['word_count'].mean():.2f}")
        
        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Character length distribution
        if label_column in self.df.columns:
            for label in self.df[label_column].unique():
                data = self.df[self.df[label_column] == label]['text_length']
                axes[0].hist(data, alpha=0.6, label=label, bins=50)
            axes[0].legend()
        else:
            axes[0].hist(self.df['text_length'], bins=50)
        
        axes[0].set_title('Text Length Distribution', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Character Length')
        axes[0].set_ylabel('Frequency')
        
        # Word count distribution
        if label_column in self.df.columns:
            for label in self.df[label_column].unique():
                data = self.df[self.df[label_column] == label]['word_count']
                axes[1].hist(data, alpha=0.6, label=label, bins=50)
            axes[1].legend()
        else:
            axes[1].hist(self.df['word_count'], bins=50)
        
        axes[1].set_title('Word Count Distribution', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Word Count')
        axes[1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('text_length_analysis.png', dpi=300, bbox_inches='tight')
        print("\n✓ Visualization saved as 'text_length_analysis.png'")
        plt.show()
        
        return text_col


# MAIN EXECUTION
if __name__ == "__main__":
    print("="*60)
    print("SPAM EMAIL DETECTION - STEP 1: DATA EXPLORATION")
    print("="*60)
    
    # Initialize loader
    # Path to your CSV file
    data_path = r"C:\Users\sayen\anaconda3\envs\spam_detection\spam_project\spam.csv"  # Update filename if different
    
    loader = SpamDataLoader(data_path)
    
    # Load data
    print("\n[1/4] Loading data...")
    df = loader.load_data()
    
    if df is not None:
        # Initial exploration
        print("\n[2/4] Performing initial exploration...")
        loader.initial_exploration()
        
        # Analyze labels
        print("\n[3/4] Analyzing label distribution...")
        label_col = loader.analyze_labels()
        
        # Text analysis
        print("\n[4/4] Analyzing text characteristics...")
        text_col = loader.text_length_analysis()
        
        print("\n" + "="*60)
        print("✓ STEP 1 COMPLETE!")
        print("="*60)
        print(f"\nKey findings:")
        print(f"  - Dataset shape: {df.shape}")
        print(f"  - Label column: {label_col}")
        print(f"  - Text column: {text_col}")
        print("\nNext step: Feature Engineering (Step 2)")
    else:
        print("\n✗ Failed to load data. Please check your file path and try again.")
        print("\nExpected file format: CSV with columns for text and labels")
        print("Example: spam.csv with 'message' and 'label' columns")