📧 Spam Detection System – Machine Learning Web Application

An end-to-end Spam Detection System built using Machine Learning and Natural Language Processing (NLP), deployed as an interactive Streamlit web application.

This project demonstrates the complete ML lifecycle — data exploration, preprocessing, feature engineering, model training, evaluation, and real-world deployment.

🚀 Live Application

🔗 Streamlit App:
https://spamproject-awdd5gdwsclbjvcrtrjwpk.streamlit.app/

📌 Project Overview

Spam emails and messages pose major challenges in communication systems.
This application classifies text as Spam or Not Spam (Ham) using multiple trained machine-learning models and provides model comparison insights through visualizations.

Key Features

NLP-based text preprocessing

Multiple ML models trained and evaluated

Saved trained models for fast inference

Interactive Streamlit UI

Cloud deployment with version-safe model loading

🧠 Machine Learning Models Used

The following models were trained and stored as serialized files (.pkl) for inference:

Model	File
Logistic Regression (SAGA)	logistic_regression_saga_model.pkl
Linear SVC (Calibrated)	linearsvc_calibrated_model.pkl
Random Forest	random_forest_model.pkl
Neural Network (MLP)	neural_network_mlp_model.pkl

Additional artifacts:

preprocessed_data.pkl → Processed dataset

model_results.pkl → Model performance metrics

🗂️ Repository Structure
spam_project/
│
├── spam_detection_ui.py          # Main Streamlit app (ENTRY POINT)
│
├── step1_data_exploration.py     # Exploratory Data Analysis (EDA)
├── step2_feature_engineering.py  # Text cleaning & feature extraction
├── step3_preprocessing.py        # Dataset preprocessing pipeline
├── step4_model_training.py       # Model training & evaluation
│
├── fix_preprocessed_data.py      # Data consistency fixes
│
├── *.pkl                         # Trained models & artifacts
│
├── model_comparison.png          # Model performance comparison
├── feature_distributions.png     # Feature distribution plots
│
├── requirements.txt              # Cloud dependencies (pip)
├── runtime.txt                   # Python version for Streamlit Cloud
│
└── README.md                     # Project documentation

🖥️ Streamlit Application (spam_detection_ui.py)

The Streamlit UI provides:

📝 Text input for spam classification

🤖 Predictions from multiple ML models

📊 Model performance comparison

📈 Stored evaluation metrics

⚡ Cached model loading for faster startup

⚙️ Local Setup (Anaconda / Conda)
1️⃣ Clone the Repository
git clone https://github.com/sayendranadh/spam_project.git
cd spam_project

2️⃣ Create and Activate Environment
conda create -n spam_env python=3.10.8 -y
conda activate spam_env

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run the Application
streamlit run spam_detection_ui.py

☁️ Cloud Deployment (Streamlit Cloud)
Required Files

requirements.txt → Python dependencies

runtime.txt → Python runtime version

runtime.txt
python-3.10.8

Deployment Steps

Push code to GitHub

Create a new app on Streamlit Cloud

Select:

Repository: spam_project

Branch: main

Main file: spam_detection_ui.py

Deploy 🚀

🔒 Model & Version Compatibility

Models trained using:

Python 3.10.8

scikit-learn 1.2.2

To avoid incompatibility issues during deployment:

scikit-learn==1.2.2


is explicitly pinned in requirements.txt.

📊 Evaluation Metrics

Accuracy

Precision

Recall

F1-Score

Performance comparison plots are included in the repository.

🛠️ Technologies Used

Python 3.10

scikit-learn

NumPy, Pandas, SciPy

NLTK, SpaCy

Streamlit

Matplotlib, Seaborn, Plotly

📈 Future Enhancements

Deep learning models (LSTM / Transformers)

Model explainability (SHAP / LIME)

REST API support

Database integration

Real-time email ingestion

👤 Author

Sayendranadh
Final Year B.Tech Student
Aspiring Data Scientist / Machine Learning Engineer

🔗 GitHub: https://github.com/sayendranadh

🔗 Live App: https://spamproject-awdd5gdwsclbjvcrtrjwpk.streamlit.app/

⭐ Acknowledgements

scikit-learn documentation

Streamlit Cloud

Open-source NLP community
