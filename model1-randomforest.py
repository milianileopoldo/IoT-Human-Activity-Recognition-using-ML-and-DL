# Imports
import pandas as pd 
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix, classification_report
from cryptography.fernet import Fernet
from io import StringIO
import numpy as np


# Utility Functions
def standardize_speed(speed):
    # Convert speed to m/min if it looks like m/s
    if speed < 25 and speed > 0:
        return speed * 60
    return speed


# Fernet Encryption and Decryption
def generate_key():
    # Creates a new Fernet Key
    return Fernet.generate_key().decode()

# Encrypt the bytes with Fernet Key
def encrypt_data(key, data_bytes):
    f = Fernet(key.encode())
    return f.encrypt(data_bytes)

# Decrypt the bytes with Fernet Key
def decrypt_data(key, encrypted_bytes):
    try:
        f = Fernet(key.encode())
        return f.decrypt(encrypted_bytes)
    except Exception:
        return None


# Calorie Estimation Function 
def estimate_calories(activity_type, moving_time, weight):
    # Calculates calories using METs (Metabolic Equivalent of Task)
    time_in_hours = moving_time / 60.0
    # METs values
    if activity_type.lower() == 'walk': mets = 3.5 
    elif activity_type.lower() == 'run': mets = 9.0 
    elif activity_type.lower() == 'ride': mets = 6.0 
    else: mets = 5.0
    return mets * weight * time_in_hours


# Overtraining Risk Check Function
def check_overtraining_risk(activity_type, moving_time):
    # Assess overtraining risk
    moving_time_minutes = moving_time 
    if activity_type in ['Run', 'Ride']:
        if moving_time_minutes > 60:
            return {
                'risk': 'Extensive Workout!',
                'message': (
                    """This was an intense session. Prioritize recovery days and ensure this fits your weekly schedule to prevent injuries."""
                )
            }
    elif activity_type == 'Walk':
        if moving_time_minutes > 120:
            return {
                'risk': 'Extensive Workout!',
                'message': (
                    """This was an intense session. Prioritize recovery days and ensure this fits your weekly schedule to prevent injuries."""
                )
            }
            
    return {'risk': 'Low Risk', 'message': 'The duration of this single activity session appears reasonable.'}


# Key Management Callback
def update_key_and_rerun():
    # Generate a new key and update the session
    new_key = generate_key()
    st.session_state.fernet_key = new_key
    st.session_state.key_verifier = new_key 


# STREAMLIT PAGE
st.set_page_config(page_title="HAR Model", page_icon="🏃", layout="wide")

st.title("🏃 Human Activity Recognition (HAR) Model")
st.write("This HAR model built with Random Forest model is your instant digital coach! Just input your distance, speed, and time, and it will immediately predict your activity (like running or cycling). Best of all, it keeps you safe by highlighting potential overtraining risks—alerting you if your workouts look too intense—so you can train smarter, not harder.")
st.markdown("---")
st.title("Activity Prediction Model")
st.write("""Welcome! This is the Activity Prediction Model interface. 
         
You can upload a dataset to train the model and generate activity predictions based on your data. 

If you want to secure your file, use the Fernet Key tools in the sidebar. Just follow the steps provided to encrypt, decrypt, and load your dataset.""")



# SIDEBAR PAGE (for optional security)

# Start session key
if 'fernet_key' not in st.session_state:
    st.session_state.fernet_key = ""

# KEY SECTION 
st.sidebar.header("KEY SECTION")
st.sidebar.markdown("""
                        1. Click **"Generate New Key"** to create a new key (if needed).
                        2. Copy the generated key.
                        3. Paste the same key when encrypting or decrypting your file.
                    """)


st.sidebar.title("Try out Symmetric Encryption!")
st.sidebar.markdown("Symmetric cryptography is a simple and fast way to lock and unlock data using a single, shared secret key. To try it out follow the steps below:")

# --- 1. Key Management Section (Sidebar) ---
st.sidebar.header("1.🔑 Generate Key ")
st.sidebar.markdown("Generate or paste your **Fernet Key** here. This key is required for encrypting your data and decrypting it.")

# The text input is initialized with the current fernet_key state.
# Its value is implicitly stored under st.session_state.key_verifier

# Key input bar
key_input_bar = st.sidebar.text_input(
    "This is your key!", 
    value=st.session_state.fernet_key,
    key="key_verifier",
    help="Paste your saved key here. Typing/pasting automatically sets the key for the current session."
)
st.session_state.fernet_key = st.session_state.key_verifier

# Generate key button
if st.sidebar.button("Generate New Key", key="gen_key_button", on_click=update_key_and_rerun):
    pass 

# Key notifications
if st.session_state.fernet_key:
    st.sidebar.info(f"**Current Active Key:** `{st.session_state.fernet_key}`")
    st.sidebar.warning(" **Reminder:** Use the SAME key for both encrypting and decrypting the file")

st.sidebar.markdown("---")

# --- 2. Data Encryption Utility (Sidebar) ---
st.sidebar.header("2. 🔒Encrypt Your Activity Dataset")
st.sidebar.markdown("Convert your original CSV into a secure, encrypted file for later use.")

# ENCRYPTION SECTION
st.sidebar.header("ENCRYPTION SECTION")
st.sidebar.markdown("""
                        1. Upload your file.
                        2. Make sure you have an encryption key.
                        3. Click **"Download Encrypted File"** to down the file.
                    """)

# Encrypting the file
if not st.session_state.fernet_key:
    st.sidebar.warning("Please enter or generate a key in Step 1 to enable encryption.")
else:
    original_file = st.sidebar.file_uploader("Upload a file to encrypt!", type=["csv"], key="encrypt_uploader")

    # Encrypted file name
    if original_file:
        original_data_bytes = original_file.getvalue()
        original_name = original_file.name
        base_name, file_ext = original_name.rsplit('.', 1) 
        encrypted_file_name = f"{base_name}_ENCRYPTED.{file_ext}"
        encrypted_data = encrypt_data(st.session_state.fernet_key, original_data_bytes)
        
        # Encrypted file download
        st.sidebar.download_button(
            label="Download Encrypted File", 
            data=encrypted_data,
            file_name=encrypted_file_name, 
            mime="application/octet-stream",
            key="encrypt_download_button_direct",
            help="Click to encrypt and download the secure version."
        )
        st.sidebar.success(f"Encrypted File: **{encrypted_file_name}**")

st.sidebar.markdown("---")

# --- 3. Encrypted File Decryption Upload (Sidebar) ---
st.sidebar.header("3. 🔓 Load Encrypted Data")
st.sidebar.markdown("Upload your **encrypted** file and use your key for decryption.")

# DECRYPTION SECTION 
st.sidebar.header("DECRYPTION SECTION")
st.sidebar.markdown("""
                        1. Upload your encrypted file.
                        2. Enter the same key used during encryption in the **KEY SECTION**.
                    """)

# Decrypt the file
encrypted_file = st.sidebar.file_uploader(
    "Upload encrypted file!", 
    type=["csv"], 
    key="decrypt_uploader",
    help="Upload the file secured in Step 2. The key from Step 1 is required for decryption."
)

st.sidebar.markdown("---")


# MAIN UPLOAD PAGE
st.header("Upload file to start prediction!")

df = None
analysis_key = st.session_state.fernet_key

# File upload (not encrypted)
plain_file = st.file_uploader("Upload file for training!", type=["csv"], key="plain_uploader")

file_to_process = encrypted_file if encrypted_file else plain_file
file_type = 'ENCRYPTED' if encrypted_file else 'plain' if plain_file else None

if file_to_process:
    # Decrypt or load CSV 
    if file_type == 'ENCRYPTED':
        if not analysis_key:
            st.error("Cannot decrypt. Enter Fernet key.")
        else:
            encrypted_bytes = file_to_process.getvalue()
            decrypted_bytes = decrypt_data(analysis_key, encrypted_bytes)
            if decrypted_bytes:
                try:
                    decrypted_string = decrypted_bytes.decode('utf-8')
                    df = pd.read_csv(StringIO(decrypted_string))
                    st.success("Encrypted file decrypted and loaded!")
                except Exception as e:
                    st.error(f"Error processing decrypted file: {e}")
                    df = None
            else:
                st.error("Decryption failed. Check key.")
    elif file_type == 'plain':
        try:
            plain_bytes = file_to_process.getvalue()
            df = pd.read_csv(StringIO(plain_bytes.decode('utf-8')))
            st.success("File loaded successfully!")
        except Exception as e:
            st.error(f"Error reading dataset: {e}")
            df = None


    # Data Cleaning and Feature Engineering 
    if df is not None and not df.empty:
        try:
            # Cleaning
            df = df.loc[:, ~df.columns.duplicated()]
            required_cols = ['distance', 'average_speed', 'max_speed', 'type']
            df = df.dropna(subset=required_cols)
            
            # Unit Standardization 
            df['average_speed'] = df['average_speed'].apply(standardize_speed)
            df['max_speed'] = df['max_speed'].apply(standardize_speed)

            # Calculates moving_time (in minutes), from distance (m), and average_speed (m/min), time (min) = (Distance (m) / Speed (m/min))
            df['moving_time'] = df.apply(
                lambda row: (row['distance'] / row['average_speed']) if row['average_speed'] > 0 else 0,
                axis=1
            )
            
            # Feature Engineering for Intensity Ration: Diffrenciates activities based on movement  
            df['intensity_ratio'] = df['average_speed'] / df['max_speed']
            df['intensity_ratio_squared'] = df['intensity_ratio'] ** 2 

            # Target Labeling
            df['activity_label'] = df['type'].apply(lambda x: x if x in ['Walk', 'Run', 'Ride'] else 'Other')
            df = df[df['activity_label'] != 'Other']

            if df['activity_label'].nunique() < 3:
                st.error("Dataset needs at least 'Walk', 'Run', and 'Ride' activities for training. Analysis halted.")
                df = None 
                
        except KeyError as e:
            st.error(f"Error: Missing required column in dataset: {e}. Please ensure your file has 'distance', 'average_speed', 'max_speed', and 'type' columns.")
            df = None
        except Exception as e:
            st.error(f"An unexpected error occurred during data preparation: {e}")
            df = None


# Model Training and Prediction
if df is not None and not df.empty:
    
    st.header("Model Training:")
    
    # Feature columns
    feature_columns = [
        'distance', 
        'moving_time',
        'average_speed', 
        'max_speed', 
        'intensity_ratio', 
        'intensity_ratio_squared'
    ]

    # Random Forest Training
    if all(col in df.columns for col in feature_columns):
        X = df[feature_columns]
        y = df['activity_label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y) 
        
        unified_classifier = RandomForestClassifier(
            n_estimators=1000, 
            random_state=42, 
            max_depth=15, 
            class_weight='balanced'
        )
        unified_classifier.fit(X_train, y_train)

        st.success("Random Forest Model trained successfully on your data!")
        st.markdown("---")
        st.header("Predict Activity & Overtraining Check")

        # Prediction Function
        def predict_activity_unified(distance, moving_time, avg_speed_input, max_speed_input):

            avg_speed = standardize_speed(avg_speed_input)
            max_speed = standardize_speed(max_speed_input)
            if max_speed == 0:
                 max_speed = 0.1
            ratio = avg_speed / max_speed
            ratio_sq = ratio ** 2
            features = pd.DataFrame([{
                'distance': distance, 
                'moving_time': moving_time,
                'average_speed': avg_speed, 
                'max_speed': max_speed, 
                'intensity_ratio': ratio, 
                'intensity_ratio_squared': ratio_sq
            }])
            
            return unified_classifier.predict(features)[0]

        # User inputs
        col_dist, col_time, col_weight = st.columns(3)
        col_avg, col_max = st.columns(2)

        with col_dist:
            # Distance in meters
            distance = st.number_input("Distance (m)", min_value=0.0, value=0.0, step=1.0, help="Total distance covered in meters (SIM_DISTANCE).")
        with col_time:
            # Moving Time in minutes 
            moving_time = st.number_input("Moving Time (min)", min_value=0.0, value=0.0, step=1.0, help="Total time spent moving in minutes (SIM_MOVING_TIME).")
        with col_weight:
            # Weight in kg for calorie calculation 
            user_weight = st.number_input("Your Weight (kg)", min_value=0.0, value=0.0, step=1.0, help="Your body weight for calorie calculation (SIM_USER_WEIGHT).")
            
        with col_avg:
            # Average speed in m/s 
            avg_speed = st.number_input("Average Speed (m/s)", min_value=0.0, value=0.0, step=1.0, help="The average speed for the activity, in meters per second (SIM_AVG_SPEED).")
        with col_max:
            # Max speed in m/s 
            max_speed = st.number_input("Max Speed (m/s)", min_value=0.0, value=0.0, step=1.0, help="The peak speed achieved, in meters per second (SIM_MAX_SPEED).")

            # Predict activity button
        if st.button("Predict Activity, Check Calories & Overtraining!"):
            
            if avg_speed > max_speed:
                st.error("Average speed cannot exceed Max speed.")
            elif avg_speed <= 0 or max_speed <= 0:
                 st.error("Speed values must be greater than zero.")
            else:

                # Prediction uses the full set of 4 user inputs
                predicted_activity = predict_activity_unified(distance, moving_time, avg_speed, max_speed)
                predicted_calories = estimate_calories(predicted_activity, moving_time, user_weight)
                
                overtraining_result = check_overtraining_risk(predicted_activity, moving_time)

                st.success(f"**Predicted Activity:** {predicted_activity}")
                st.info(f"**Estimated Calories Burned:** {predicted_calories:.2f} kcal")

                if overtraining_result['risk'] != 'Low Risk':
                    st.warning(f"**Overtraining Risk Check:** {overtraining_result['risk']}")
                    st.markdown(f"**Observation:** {overtraining_result['message']}")
                else:
                    st.info(f"**Overtraining Risk Check:** {overtraining_result['risk']}")
                    st.markdown(f"**Observation:** {overtraining_result['message']}")


                # Evaluation
                st.markdown("---")
                st.header("Model Evaluation Results")
                
                y_pred_unified = unified_classifier.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred_unified)
                st.write(f"**Overall 3-Class Accuracy (Walk, Run, Ride):** {accuracy * 100:.2f}%")
                
                # Metrics
                st.subheader("Detailed Classification Metrics")
                
                classes = sorted(y_test.unique())
                report = classification_report(y_test, y_pred_unified, target_names=classes, output_dict=True, zero_division=0)
                report_df = pd.DataFrame(report).transpose()
                
                metrics_to_format = ['precision', 'recall', 'f1-score']
                for col in metrics_to_format:
                    if col in report_df.columns:
                        report_df[col] = (report_df[col] * 100).apply(lambda x: f"{x:.2f}%" if pd.notna(x) else x)

                st.dataframe(report_df, use_container_width=True)
                
                st.markdown("---")
                
                # Visualizations
                col_cm, col_imp = st.columns(2)
                
                with col_cm:
                    st.subheader("Confusion Matrix:")
                    
                    cm = confusion_matrix(y_test, y_pred_unified, labels=classes)
                    fig, ax = plt.subplots(figsize=(6, 6))
                    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
                    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
                    
                    ax.set_title("Unified HAR Classifier Results", fontsize=14)
                    ax.set_xlabel("Predicted Label", fontsize=12)
                    ax.set_ylabel("True Label", fontsize=12)

                    st.pyplot(fig)
                
                with col_imp:
                    st.subheader("Feature Importance:")
                    
                    importances = unified_classifier.feature_importances_
                    feature_names = X_train.columns
                    
                    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
                    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=True)

                    fig_imp, ax_imp = plt.subplots(figsize=(8, 5))
                    ax_imp.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='teal')
                    ax_imp.set_xlabel("Feature Importance Score (Random Forest)", fontsize=12)
                    ax_imp.set_title("Feature Importance in Unified HAR Model", fontsize=14)
                    
                    st.pyplot(fig_imp)
                    
    else:
        st.error("Model training failed: Critical features required for training are missing from the processed dataset.")
