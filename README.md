Activity Prediction App - Random Forest

This Streamlit application predicts whether an activity is a walk, run, or ride based on the trained dataset with Random Forest model.

It also provides estimated calories burned, an overtraining risk check, and an optional cryptography feature that lets users encrypt or decrypt datasets using Fernet (symmetric encryption).

--------------------------------------------------------------------------------------------------------------------
FEATURES:

1. Activity Prediction (Random Forest Model)

2. Health Insights 
    > Estimated Calories Burned using MET calculations
    > Overtraining Risk Check 
    
3. Cryptography feature
    > Encrypt and decrypt CSV files using Fernet symmetric encryption
    > Allows secure dataset handling
    
-------------------------------------------------------------------------------------------------------------------

How to run the App?

1. Installation:

pip install -r streamlit

2. Launching

streamlit run IOT1.py

--------------------------------------------------------------------------------------------------------------------

Guide on how to use the app:

1. Upload your dataset
    > Run the app with "streamlit run IOT1.py"
    > In the Upload Dataset section, click Browse and select your CSV file.
    > Once uploaded, the app confirms the file was loaded successfully

2. Train the Random Forest model
    > After the dataset loads, the model trains automatically
    > When training is complete, a success message will appear stating that the model is ready.

3. Test the model with inputted data
    > enter:
            Distance (m)
            Moving time (min)
            Weight (kg)
            Average speed (m/s)
            Max speed (m/s)
    > Then click predict
    > The app will display results. 

4. (OPTIONAL) Secure dataset file with encryption
    > Open the tab on the top left for the Key & Encryption Section
    > Generate a key 
    > Upload a dataset to encrypt
    > Download the encrypted file
    > To decrypt later, upload the encrypted file in the main section and provide the same key.

    TAKE NOTE: If the wrong key is used, the file will not load as encryption is working.

-------------------------------------------------------------------------------------------------------------------






