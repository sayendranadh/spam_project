#!/usr/bin/env python3
"""
SPAM EMAIL DETECTION - STEP 4: MODEL TRAINING
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_auc_score,
    roc_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

def print_header():
    """Print script header"""
    print("=" * 60)
    print("SPAM EMAIL DETECTION - STEP 4: MODEL TRAINING")
    print("=" * 60)
    print()

def load_preprocessed_data():
    """Load preprocessed data"""
    print("Loading preprocessed data...")
    
    # Try different possible file paths
    possible_paths = [
        'preprocessed_data.pkl',
        './preprocessed_data.pkl',
        r'C:\Users\sayen\anaconda3\envs\spam_detection\spam_project\preprocessed_data.pkl'
    ]
    
    for file_path in possible_paths:
        try:
            print(f"   Trying: {file_path}")
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            print("✓ Preprocessed data loaded successfully!")
            print(f"   Loaded from: {file_path}")
            print(f"   Available keys: {list(data.keys())}")
            return data
            
        except FileNotFoundError:
            print(f"   ❌ Not found: {file_path}")
            continue
        except Exception as e:
            print(f"   ❌ Error loading {file_path}: {str(e)}")
            continue
    
    # If all paths fail, ask user for manual path
    print("\n❌ Could not find preprocessed_data.pkl in any expected location!")
    print("Please provide the full path to your preprocessed_data.pkl file:")
    
    # Try to get user input for file path
    try:
        user_path = input("Enter full file path: ").strip().strip('"').strip("'")
        if user_path and os.path.exists(user_path):
            with open(user_path, 'rb') as f:
                data = pickle.load(f)
            print("✓ Data loaded successfully from user-provided path!")
            print(f"   Available keys: {list(data.keys())}")
            return data
        else:
            print("❌ Invalid path or file not found!")
    except:
        print("❌ Failed to get user input or load file!")
    
    return None

def print_data_summary(data):
    """Print data summary"""
    print("\n" + "=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)
    
    print(f"Training samples: {data['X_train_tfidf'].shape[0]}")
    print(f"Validation samples: {data['X_val_tfidf'].shape[0]}")
    print(f"Test samples: {data['X_test_tfidf'].shape[0]}")
    print(f"TF-IDF features: {data['X_train_tfidf'].shape[1]}")
    print(f"Engineered features: {data['X_features_train_scaled'].shape[1]}")
    
    # Label distribution
    print(f"\nTraining label distribution:")
    unique, counts = np.unique(data['y_train'], return_counts=True)
    for label, count in zip(unique, counts):
        label_name = 'Ham' if label == 0 else 'Spam'
        print(f"   {label_name}: {count} ({count/len(data['y_train'])*100:.1f}%)")

def combine_features(tfidf_features, engineered_features):
    """Combine TF-IDF and engineered features"""
    from scipy.sparse import hstack, csr_matrix
    
    # Convert engineered features to sparse matrix
    eng_sparse = csr_matrix(engineered_features)
    
    # Combine features
    combined = hstack([tfidf_features, eng_sparse])
    
    return combined

def train_models(data):
    """Train multiple models"""
    print("\n" + "=" * 60)
    print("MODEL TRAINING")
    print("=" * 60)
    
    # Combine features
    print("\nCombining TF-IDF and engineered features...")
    X_train = combine_features(data['X_train_tfidf'], data['X_features_train_scaled'])
    X_val = combine_features(data['X_val_tfidf'], data['X_features_val_scaled'])
    X_test = combine_features(data['X_test_tfidf'], data['X_features_test_scaled'])
    
    print(f"✓ Combined feature shape: {X_train.shape}")
    
    y_train = data['y_train']
    y_val = data['y_val']
    y_test = data['y_test']
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,   # use all CPU cores
            verbose=1    # show training progress
        ),
        'Naive Bayes': MultinomialNB(),
        'SVM': SVC(probability=True, random_state=42),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"\n[{list(models.keys()).index(name) + 1}/{len(models)}] Training {name}...")
        
        try:
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred)
            recall = recall_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred)
            auc = roc_auc_score(y_val, y_pred_proba)
            
            # Cross-validation score
            # (Keep default n_jobs here; RF already parallelizes internally)
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            
            results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc': auc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            trained_models[name] = model
            
            print(f"   ✓ Accuracy: {accuracy:.4f}")
            print(f"   ✓ F1-Score: {f1:.4f}")
            print(f"   ✓ AUC: {auc:.4f}")
            
        except Exception as e:
            print(f"   ❌ Error training {name}: {str(e)}")
    
    return results, trained_models, (X_train, X_val, X_test, y_train, y_val, y_test)

def display_results(results):
    """Display model comparison results"""
    print("\n" + "=" * 60)
    print("MODEL COMPARISON RESULTS")
    print("=" * 60)
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        name: {
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1_score'],
            'AUC': metrics['auc'],
            'CV Mean': metrics['cv_mean'],
            'CV Std': metrics['cv_std']
        }
        for name, metrics in results.items()
    }).T
    
    # Round to 4 decimal places
    results_df = results_df.round(4)
    
    print("\n📊 Performance Summary:")
    print(results_df)
    
    # Find best models
    best_accuracy = results_df['Accuracy'].idxmax()
    best_f1 = results_df['F1-Score'].idxmax()
    best_auc = results_df['AUC'].idxmax()
    
    print(f"\n🏆 Best Models:")
    print(f"   Best Accuracy: {best_accuracy} ({results_df.loc[best_accuracy, 'Accuracy']:.4f})")
    print(f"   Best F1-Score: {best_f1} ({results_df.loc[best_f1, 'F1-Score']:.4f})")
    print(f"   Best AUC: {best_auc} ({results_df.loc[best_auc, 'AUC']:.4f})")
    
    return results_df

def create_visualizations(results, data_splits):
    """Create model performance visualizations"""
    print("\nGenerating visualizations...")
    
    X_train, X_val, X_test, y_train, y_val, y_test = data_splits
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Model Performance Analysis', fontsize=16, fontweight='bold')
    
    # 1. Performance Metrics Comparison
    metrics_data = []
    for name, metrics in results.items():
        metrics_data.append({
            'Model': name,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1_score']
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.set_index('Model')[['Accuracy', 'Precision', 'Recall', 'F1-Score']].plot(
        kind='bar', ax=axes[0,0], rot=45
    )
    axes[0,0].set_title('Performance Metrics Comparison')
    axes[0,0].set_ylabel('Score')
    axes[0,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. AUC Scores
    auc_scores = [metrics['auc'] for metrics in results.values()]
    model_names = list(results.keys())
    
    bars = axes[0,1].bar(model_names, auc_scores, color='skyblue', alpha=0.7)
    axes[0,1].set_title('AUC Scores Comparison')
    axes[0,1].set_ylabel('AUC Score')
    axes[0,1].tick_params(axis='x', rotation=45)
    axes[0,1].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, score in zip(bars, auc_scores):
        axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                      f'{score:.3f}', ha='center', va='bottom')
    
    # 3. ROC Curves
    for name, metrics in results.items():
        fpr, tpr, _ = roc_curve(y_val, metrics['probabilities'])
        axes[0,2].plot(fpr, tpr, label=f'{name} (AUC = {metrics["auc"]:.3f})')
    
    axes[0,2].plot([0, 1], [0, 1], 'k--', label='Random')
    axes[0,2].set_xlabel('False Positive Rate')
    axes[0,2].set_ylabel('True Positive Rate')
    axes[0,2].set_title('ROC Curves')
    axes[0,2].legend()
    axes[0,2].grid(True, alpha=0.3)
    
    # 4. Cross-Validation Scores
    cv_means = [metrics['cv_mean'] for metrics in results.values()]
    cv_stds = [metrics['cv_std'] for metrics in results.values()]
    
    bars = axes[1,0].bar(model_names, cv_means, yerr=cv_stds, 
                        color='lightcoral', alpha=0.7, capsize=5)
    axes[1,0].set_title('Cross-Validation Scores')
    axes[1,0].set_ylabel('CV Accuracy')
    axes[1,0].tick_params(axis='x', rotation=45)
    axes[1,0].grid(True, alpha=0.3)
    
    # Add value labels
    for bar, mean, std in zip(bars, cv_means, cv_stds):
        axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.005,
                      f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 5. Best Model Confusion Matrix (using model with highest F1-score)
    best_model_name = max(results.keys(), key=lambda x: results[x]['f1_score'])
    best_predictions = results[best_model_name]['predictions']
    
    cm = confusion_matrix(y_val, best_predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1,1],
                xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    axes[1,1].set_title(f'Confusion Matrix - {best_model_name}')
    axes[1,1].set_ylabel('True Label')
    axes[1,1].set_xlabel('Predicted Label')
    
    # 6. Model Ranking
    ranking_metrics = ['accuracy', 'f1_score', 'auc']
    ranking_data = []
    
    for metric in ranking_metrics:
        sorted_models = sorted(results.items(), key=lambda x: x[1][metric], reverse=True)
        for rank, (name, _) in enumerate(sorted_models, 1):
            ranking_data.append({
                'Model': name,
                'Metric': metric.replace('_', ' ').title(),
                'Rank': rank
            })
    
    ranking_df = pd.DataFrame(ranking_data)
    ranking_pivot = ranking_df.pivot(index='Model', columns='Metric', values='Rank')
    
    sns.heatmap(ranking_pivot, annot=True, fmt='d', cmap='RdYlGn_r', ax=axes[1,2],
                cbar_kws={'label': 'Rank (1 = Best)'})
    axes[1,2].set_title('Model Ranking by Metrics')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Visualization saved as 'model_comparison.png'")
    
    return fig

def evaluate_best_model(results, trained_models, data_splits):
    """Evaluate the best model on test set"""
    print("\n" + "=" * 60)
    print("BEST MODEL EVALUATION")
    print("=" * 60)
    
    X_train, X_val, X_test, y_train, y_val, y_test = data_splits
    
    # Find best model based on F1-score
    best_model_name = max(results.keys(), key=lambda x: results[x]['f1_score'])
    best_model = trained_models[best_model_name]
    
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"   Validation F1-Score: {results[best_model_name]['f1_score']:.4f}")
    
    # Evaluate on test set
    print("\n📊 Test Set Evaluation:")
    y_test_pred = best_model.predict(X_test)
    y_test_proba = best_model.predict_proba(X_test)[:, 1]
    
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_proba)
    
    print(f"   Accuracy:  {test_accuracy:.4f}")
    print(f"   Precision: {test_precision:.4f}")
    print(f"   Recall:    {test_recall:.4f}")
    print(f"   F1-Score:  {test_f1:.4f}")
    print(f"   AUC:       {test_auc:.4f}")
    
    # Detailed classification report
    print("\n📋 Detailed Classification Report:")
    print(classification_report(y_test, y_test_pred, target_names=['Ham', 'Spam']))
    
    return best_model_name, best_model, {
        'accuracy': test_accuracy,
        'precision': test_precision,
        'recall': test_recall,
        'f1_score': test_f1,
        'auc': test_auc
    }

def save_models(trained_models, results, best_model_name):
    """Save trained models and results"""
    print("\n" + "=" * 60)
    print("SAVING MODELS AND RESULTS")
    print("=" * 60)
    
    # Save all models
    print("\nSaving trained models...")
    models_dir = 'trained_models'
    os.makedirs(models_dir, exist_ok=True)
    
    for name, model in trained_models.items():
        filename = f"{models_dir}/{name.lower().replace(' ', '_')}_model.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        print(f"   ✓ {name} saved to {filename}")
    
    # Save best model separately
    best_model = trained_models[best_model_name]
    with open('best_spam_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    print(f"   ✓ Best model ({best_model_name}) saved to 'best_spam_model.pkl'")
    
    # Save results
    with open('model_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print("   ✓ Results saved to 'model_results.pkl'")
    
    # Save results as CSV
    results_df = pd.DataFrame({
        name: {
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1_score'],
            'AUC': metrics['auc'],
            'CV_Mean': metrics['cv_mean'],
            'CV_Std': metrics['cv_std']
        }
        for name, metrics in results.items()
    }).T
    
    results_df.to_csv('model_results.csv')
    print("   ✓ Results saved to 'model_results.csv'")

def print_completion_summary(best_model_name, test_metrics):
    """Print completion summary"""
    print("\n" + "=" * 60)
    print("✓ STEP 4 COMPLETE!")
    print("=" * 60)
    
    print(f"\n🎯 Best Model: {best_model_name}")
    print(f"   Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"   Test F1-Score: {test_metrics['f1_score']:.4f}")
    print(f"   Test AUC: {test_metrics['auc']:.4f}")
    
    print(f"\n📁 Generated Files:")
    print(f"   - model_comparison.png (visualizations)")
    print(f"   - trained_models/ (all models)")
    print(f"   - best_spam_model.pkl (best model)")
    print(f"   - model_results.pkl (detailed results)")
    print(f"   - model_results.csv (results summary)")
    
    print(f"\n✨ Next step: Model Deployment (Step 5)")
    print(f"   - Create prediction interface")
    print(f"   - Build web application")
    print(f"   - API development")

def main():
    """Main execution function"""
    print_header()
    
    # Print current working directory for debugging
    print(f"Current working directory: {os.getcwd()}")
    print(f"Files in current directory: {os.listdir('.')}")
    print()
    
    # Load data
    data = load_preprocessed_data()
    if data is None:
        print("\n❌ Failed to load preprocessed data!")
        print("Please ensure preprocessed_data.pkl exists and try again.")
        return
    
    # Print data summary
    print_data_summary(data)
    
    # Train models
    results, trained_models, data_splits = train_models(data)
    
    if not results:
        print("❌ No models trained successfully!")
        return
    
    # Display results
    results_df = display_results(results)
    
    # Create visualizations
    create_visualizations(results, data_splits)
    
    # Evaluate best model
    best_model_name, best_model, test_metrics = evaluate_best_model(
        results, trained_models, data_splits
    )
    
    # Save models and results
    save_models(trained_models, results, best_model_name)
    
    # Print completion summary
    print_completion_summary(best_model_name, test_metrics)

if __name__ == "__main__":
    main()
