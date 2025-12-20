# 📧 Spam Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10.8-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2.2-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**An end-to-end Machine Learning web application for intelligent spam detection**

[🚀 Live Demo](https://spamproject-awdd5gdwsclbjvcrtrjwpk.streamlit.app/) • [📖 Documentation](#documentation) • [🐛 Report Bug](https://github.com/sayendranadh/spam_project/issues) • [✨ Request Feature](https://github.com/sayendranadh/spam_project/issues)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Live Demo](#-live-demo)
- [Machine Learning Models](#-machine-learning-models)
- [Technology Stack](#-technology-stack)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Model Performance](#-model-performance)
- [Deployment](#-deployment)
- [Future Enhancements](#-future-enhancements)
- [Contributing](#-contributing)
- [Author](#-author)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)

---

## 🎯 Overview

The **Spam Detection System** is a comprehensive machine learning application that classifies text messages and emails as either **Spam** or **Ham** (legitimate). Built with a focus on demonstrating the complete ML lifecycle, this project encompasses:

- 📊 **Data Exploration & Analysis**
- 🧹 **Text Preprocessing & NLP**
- 🔧 **Feature Engineering**
- 🤖 **Model Training & Evaluation**
- 🚀 **Production Deployment**

This application showcases real-world implementation of multiple ML algorithms with an interactive web interface, making it ideal for understanding practical machine learning workflows.

---

## ✨ Features

### Core Functionality
- 🎯 **Real-time Spam Classification** - Instant predictions on user-input text
- 🔄 **Multiple ML Models** - Compare predictions from 4 different algorithms
- 📊 **Performance Metrics** - Comprehensive evaluation with accuracy, precision, recall, and F1-score
- 📈 **Visual Analytics** - Interactive charts for model comparison and feature distributions
- ⚡ **Optimized Performance** - Cached model loading for lightning-fast responses

### Technical Highlights
- 🧠 Advanced NLP preprocessing pipeline
- 💾 Serialized model persistence for efficient deployment
- 🎨 Clean, intuitive Streamlit UI
- ☁️ Cloud-ready with version-controlled dependencies
- 🔒 Production-grade error handling and validation

---

## 🚀 Live Demo

Experience the application in action:

**🔗 [Streamlit Cloud Deployment](https://spamproject-awdd5gdwsclbjvcrtrjwpk.streamlit.app/)**

### Quick Test Examples

Try these sample inputs:

**Spam Example:**
```
URGENT! You have won $1,000,000! Click here to claim your prize NOW!
```

**Ham Example:**
```
Hey, are we still meeting for coffee tomorrow at 3pm?
```

---

## 🧠 Machine Learning Models

The system employs **four distinct machine learning models**, each optimized for text classification:

| Model | Algorithm | File | Key Strength |
|-------|-----------|------|--------------|
| **Logistic Regression** | SAGA Solver | `logistic_regression_saga_model.pkl` | Fast training, interpretable coefficients |
| **Linear SVC** | Calibrated Classifier | `linearsvc_calibrated_model.pkl` | Excellent for high-dimensional text data |
| **Random Forest** | Ensemble Method | `random_forest_model.pkl` | Robust to overfitting, feature importance |
| **Neural Network** | MLP Classifier | `neural_network_mlp_model.pkl` | Captures complex non-linear patterns |

### Additional Artifacts

- `preprocessed_data.pkl` - Cleaned and processed training dataset
- `model_results.pkl` - Performance metrics for all models
- `feature_distributions.png` - Visualization of feature importance
- `model_comparison.png` - Comparative analysis charts

---

## 🛠️ Technology Stack

### Core Technologies

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.10.8 |
| **ML Framework** | scikit-learn 1.2.2 |
| **NLP** | NLTK, SpaCy |
| **Data Processing** | NumPy, Pandas, SciPy |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Web Framework** | Streamlit |
| **Deployment** | Streamlit Cloud |

---

## 💻 Installation

### Prerequisites

- Python 3.10.8
- Anaconda/Miniconda (recommended) or pip
- Git

### Option 1: Using Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/sayendranadh/spam_project.git
cd spam_project

# Create a new conda environment
conda create -n spam_env python=3.10.8 -y

# Activate the environment
conda activate spam_env

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Using pip & venv

```bash
# Clone the repository
git clone https://github.com/sayendranadh/spam_project.git
cd spam_project

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🎮 Usage

### Running Locally

```bash
# Ensure your environment is activated
conda activate spam_env  # or: source venv/bin/activate

# Launch the Streamlit application
streamlit run spam_detection_ui.py
```

The application will open automatically in your default browser at `http://localhost:8501`

### Using the Application

1. **Enter Text**: Type or paste the message you want to classify
2. **Select Model**: Choose from the available ML models (or compare all)
3. **Get Prediction**: Click "Classify" to see results
4. **View Metrics**: Explore performance statistics and visualizations

### Training Pipeline (Optional)

To retrain models with your own data:

```bash
# Step 1: Exploratory Data Analysis
python step1_data_exploration.py

# Step 2: Feature Engineering
python step2_feature_engineering.py

# Step 3: Data Preprocessing
python step3_preprocessing.py

# Step 4: Model Training
python step4_model_training.py

# Run the UI with new models
streamlit run spam_detection_ui.py
```

---

## 📁 Project Structure

```
spam_project/
│
├── 📄 spam_detection_ui.py              # Main Streamlit application (ENTRY POINT)
│
├── 🔬 ML Pipeline Scripts
│   ├── step1_data_exploration.py        # EDA and data insights
│   ├── step2_feature_engineering.py     # Text preprocessing & feature extraction
│   ├── step3_preprocessing.py           # Data cleaning pipeline
│   └── step4_model_training.py          # Model training & evaluation
│
├── 🔧 Utilities
│   └── fix_preprocessed_data.py         # Data consistency fixes
│
├── 💾 Model Artifacts (*.pkl)
│   ├── logistic_regression_saga_model.pkl
│   ├── linearsvc_calibrated_model.pkl
│   ├── random_forest_model.pkl
│   ├── neural_network_mlp_model.pkl
│   ├── preprocessed_data.pkl
│   └── model_results.pkl
│
├── 📊 Visualizations
│   ├── model_comparison.png             # Model performance charts
│   └── feature_distributions.png        # Feature importance plots
│
├── ⚙️ Configuration Files
│   ├── requirements.txt                 # Python dependencies
│   └── runtime.txt                      # Python version for deployment
│
└── 📖 README.md                         # Project documentation
```

---

## 📊 Model Performance

### Evaluation Metrics

All models are evaluated using standard classification metrics:

- **Accuracy** - Overall prediction correctness
- **Precision** - Spam prediction reliability
- **Recall** - Spam detection coverage
- **F1-Score** - Harmonic mean of precision and recall

### Comparative Analysis

View the `model_comparison.png` file for detailed performance visualizations showing:
- Model accuracy comparison
- Precision-Recall trade-offs
- Confusion matrices
- ROC curves

---

## ☁️ Deployment

### Streamlit Cloud Deployment

#### Required Files

- `requirements.txt` - All Python dependencies
- `runtime.txt` - Python version specification

**runtime.txt**
```
python-3.10.8
```

#### Deployment Steps

1. **Prepare Repository**
   ```bash
   git add .
   git commit -m "Deploy to Streamlit Cloud"
   git push origin main
   ```

2. **Configure Streamlit Cloud**
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Select your repository: `spam_project`
   - Branch: `main`
   - Main file: `spam_detection_ui.py`

3. **Deploy**
   - Click "Deploy!"
   - Wait for build completion
   - Your app will be live at: `https://[app-name].streamlit.app/`

### Version Compatibility

⚠️ **Important**: Models are trained with specific library versions

```txt
Python: 3.10.8
scikit-learn: 1.2.2
```

Using different versions may cause compatibility issues. The `requirements.txt` file pins exact versions to ensure consistency.

---

## 🔮 Future Enhancements

### Planned Features

- [ ] **Deep Learning Models**
  - LSTM networks for sequential text analysis
  - Transformer-based models (BERT, DistilBERT)
  
- [ ] **Model Interpretability**
  - SHAP values for feature importance
  - LIME for local interpretability
  
- [ ] **API Development**
  - RESTful API with FastAPI
  - Swagger/OpenAPI documentation
  
- [ ] **Data Management**
  - Database integration (PostgreSQL/MongoDB)
  - User feedback collection system
  
- [ ] **Advanced Features**
  - Real-time email ingestion
  - Batch processing capabilities
  - Multi-language support
  - Custom model training interface

### Contributions Welcome!

Have ideas for improvements? Check out the [Contributing](#-contributing) section below.

---

## 🤝 Contributing

Contributions are what make the open-source community amazing! Any contributions you make are **greatly appreciated**.

### How to Contribute

1. **Fork the Project**
   ```bash
   git clone https://github.com/sayendranadh/spam_project.git
   ```

2. **Create a Feature Branch**
   ```bash
   git checkout -b feature/AmazingFeature
   ```

3. **Commit Changes**
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```

4. **Push to Branch**
   ```bash
   git push origin feature/AmazingFeature
   ```

5. **Open a Pull Request**

### Contribution Guidelines

- Write clear, descriptive commit messages
- Follow PEP 8 style guidelines for Python code
- Add tests for new features
- Update documentation as needed
- Ensure all tests pass before submitting PR

---

## 👤 Author

**Sayendranadh**

- 🎓 Final Year B.Tech Student
- 💼 Aspiring Data Scientist / Machine Learning Engineer
- 🌐 GitHub: [@sayendranadh](https://github.com/sayendranadh)
- 🔗 LinkedIn: [Connect with me](https://www.linkedin.com/in/veduruvada-satya-sayendranadh-320296260/)
- 📧 Email: sayendranadh2005@gmail.com

### Connect & Support

If you find this project helpful:
- ⭐ Star this repository
- 🐦 Share it with others
- 🤝 Connect on [LinkedIn](https://www.linkedin.com/in/veduruvada-satya-sayendranadh-320296260/)

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### MIT License Summary

```
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software.
```

---

## 🙏 Acknowledgements

Special thanks to:

- [scikit-learn](https://scikit-learn.org/) - Comprehensive ML library and documentation
- [Streamlit](https://streamlit.io/) - Amazing framework for ML web apps
- [Streamlit Cloud](https://streamlit.io/cloud) - Free hosting for data apps
- [NLTK](https://www.nltk.org/) - Natural Language Toolkit
- [SpaCy](https://spacy.io/) - Industrial-strength NLP
- Open-source NLP community for datasets and research

### Inspiration & Resources

- [Kaggle Spam Classification Datasets](https://www.kaggle.com/)
- [scikit-learn Text Classification Tutorial](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

<div align="center">

**⭐ Star this repository if you found it helpful!**

Made with ❤️ by [Sayendranadh](https://github.com/sayendranadh)

[🔝 Back to Top](#-spam-detection-system)

</div>
