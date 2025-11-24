# Activity Prediction & Health Insights Web App

## Our HAR Model 1 addresses the need for accessible and secure activity analysis. 

The application leverages a Random Forest model within a user-friendly Streamlit interface to predict physical activities like Walking, Running, and Riding. Beyond simple classification, it offers insights into health metrics and features an integrated cryptography system. A custom GUI provides a clear, visual comparison between encrypted and decrypted data, effectively educating users on the practical benefits and necessity of data security.

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

## 🌐 Live Demo

Experience the application firsthand by visiting our live deployment:

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](http://172.18.0.177:8501)

## ✨ Features

- **🤖 Activity Prediction**: Classifies activities (Walk, Run, Ride) using a trained Random Forest model
- **💪 Health Insights**:
  - **Calorie Estimation**: Calculates estimated calories burned based on activity type and user metrics
  - **Overtraining Risk Check**: Alerts users to potential overtraining based on activity frequency and intensity
- **🔐 Cryptography Suite**:
  - **Encrypt/Decrypt CSV Files**: Securely handle datasets using Fernet symmetric encryption
  - **Key Generation**: Generate and manage encryption keys within the app

## 🚀 Quick Start

### Prerequisites

- Python 3.7+
- pip (Python package manager)

### Installation & Launch

1. **Clone or download the project files** to your local machine

2. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
  Note: If you don't have a requirements.txt, install the packages manually:

bash
pip install streamlit pandas scikit-learn cryptography
Run the application:

bash
streamlit run IOT1.py
Your default web browser will automatically open and display the app

# 📖 How to Use
## Activity Prediction
### Upload Your Dataset:
In the sidebar, go to the "Upload Dataset" section
Click "Browse files" and select your CSV file containing activity data
A success message will confirm the dataset has been loaded

### Train the Model:
The Random Forest model trains automatically upon a successful dataset upload
Wait for the "Model training complete!" message

### Make a Prediction:
In the main panel, enter the following metrics:
1. Distance (m)
2. Moving Time (min)
3. Weight (kg)
4. Average Speed (m/s)
5. Max Speed (m/s)

### Click the "Predict Activity" button
The app will display the predicted activity and health insights

## 2. Cryptography Feature (Optional)
This feature allows you to encrypt and decrypt your dataset files for secure storage and transfer

### 1. Generate a Key:
Navigate to the "Key & Encryption Section" in the sidebar
Click "Generate New Key". Save this key securely—you will need it to decrypt the file later

### Encrypt a File:
Under "Encrypt a Dataset", upload a CSV file
Paste the generated key into the input field
Click "Encrypt File"
Download the resulting encrypted file (.encrypted)

### Decrypt a File:
Under "Decrypt a Dataset", upload an encrypted file (.encrypted)
Provide the same key used for encryption
Click "Decrypt File"
The decrypted file will be loaded for use in the prediction module

> ⚠️ Important Note: If an incorrect key is used for decryption, the process will fail, ensuring your data's security

## 🗂️ Project Structure
text
- model1-randomforest.py   # Main Streamlit application script
- README.md           # This file

### 🔧 Technical Details
1. Model: Random Forest Classifier from scikit-learn
2. Encryption: Fernet (symmetric encryption) from the cryptography library
3. Calorie Calculation: Based on Metabolic Equivalent of Task (MET) values

### 📝 License
This project is for educational purposes

