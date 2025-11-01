Data Science Beginners Project
# Customer Churn Prediction App

A **Machine Learning web app** built with **Streamlit** that predicts whether a customer is likely to churn based on their account information and service usage.

## Project Overview

Customer churn is one of the key challenges faced by subscription-based businesses.  
This project aims to predict whether a customer will **churn** or **stay**, using a trained classification model.  
It helps companies make data-driven retention strategies.

## Features

Interactive web app built using **Streamlit**  
Machine learning model trained on customer churn dataset  
Probability-based prediction (confidence level)  
Displays **Top 10 important features** influencing churn  
User-friendly UI with live input and visualization  

## Tech Stack

| Component | Technology |
|------------|-------------|
| Language | Python |
| Web Framework | Streamlit |
| Machine Learning | Scikit-learn |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib |
| Model Serialization | Joblib |

## Project Structure

Customer-Churn-Prediction/
│
├── app.py                   # Streamlit web app
├── churn_model.pkl          # Trained ML model
├── model_columns.pkl        # Model feature columns
├── requirements.txt         # Dependencies list
├── README.md                # Documentation
└── dataset.csv              # link : https://www.kaggle.com/datasets/blastchar/telco-customer-churn

## Installation & Setup

Run the following commands in your terminal

# Clone this repository
git clone https://github.com/AaryaMehta2506/Customer-Churn-Prediction.git

# Navigate to the project folder
cd Customer-Churn-Prediction

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate   # for Windows
source venv/bin/activate  # for Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py

Once the server starts, open the app in your browser at:
http://localhost:8501

## Model Details

- **Algorithm Used:** Random Forest Classifier  
- **Accuracy:** ~78%  
- **Target Variable:** `Churn` (Yes/No)  
- **Preprocessing:** Label Encoding + One-Hot Encoding  
- **Tools Used:** Scikit-learn, Pandas, NumPy

## Deploy on Streamlit Cloud

You can deploy your app easily using **Streamlit Cloud**:

1. Push your code to GitHub  
2. Go to https://share.streamlit.io  
3. Connect your GitHub repo  
4. Set the file path to:
   app.py  
5. Click **Deploy**
Done! Your app will be live in minutes

## Contributing
Contributions are welcome!
Feel free to fork the repository, improve the game, and open a pull request. Let's grow this classic game together!

## License
This project is licensed under the [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)

## Author
**Aarya Mehta**  
🔗 [GitHub Profile](https://github.com/AaryaMehta2506)


