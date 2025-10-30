# --- Imports ---
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix, classification_report
from cryptography.fernet import Fernet
from io import StringIO
import numpy as np

# --- Utility Functions (Kaggle Alignment) ---

def standardize_speed(speed):
    """
    KAGGLE ALIGNMENT: Standardizes speed units. 
    If the speed is low (assumed to be m/s, e.g., < 25), converts it to m/min (m/s * 60).
    The model trains on m/min units for speed features.
    """
    # A speed below 25 m/s (90 km/h) is realistically in m/s for running/walking/biking data
    if speed < 25 and speed > 0:
        return speed * 60
    return speed

# --- Fernet Encryption/Decryption Functions (Data Focus) ---
def generate_key():
    """Generates a new Fernet key."""
    return Fernet.generate_key().decode()

def encrypt_data(key, data_bytes):
    """Encrypts raw bytes data using Fernet."""
    f = Fernet(key.encode())
    return f.encrypt(data_bytes)

def decrypt_data(key, encrypted_bytes):
    """Decrypts Fernet-encrypted bytes data."""
    try:
        f = Fernet(key.encode())
        return f.decrypt(encrypted_bytes)
    except Exception:
        # Catch decryption errors (wrong key, corrupted data)
        return None

# --- Calorie Estimation Function ---
def estimate_calories(activity_type, moving_time, weight):
    """
    Calculates estimated calories burned using standard METs values (METs * Wt * Time_Hours).
    moving_time is expected in minutes.
    """
    time_in_hours = moving_time / 60.0
    
    # METs (Metabolic Equivalent of Task) values
    if activity_type.lower() == 'walk': mets = 3.5 
    elif activity_type.lower() == 'run': mets = 9.0 
    elif activity_type.lower() == 'ride': mets = 6.0 
    else: mets = 5.0 # Default MET
        
    return mets * weight * time_in_hours

# --- Overtraining Risk Check Function ---
def check_overtraining_risk(activity_type, moving_time):
    """
    Assesses a single activity session for high volume.
    moving_time is expected in minutes.
    """
    moving_time_minutes = moving_time 

    if activity_type in ['Run', 'Ride']:
        if moving_time_minutes > 60:
            return {
                'risk': 'High Volume / Vigorous Alert',
                'message': (
                    "This is a vigorous activity session lasting over an hour. "
                    "Ensure this fits your weekly schedule and prioritize recovery days "
                    "to prevent potential overreaching."
                )
            }
    elif activity_type == 'Walk':
        if moving_time_minutes > 120:
            return {
                'risk': 'Long Duration Alert',
                'message': (
                    "This walking session exceeds two hours. While low impact, "
                    "extended duration can lead to fatigue or joint stress. "
                    "Listen to your body and ensure adequate rest."
                )
            }
            
    return {'risk': 'Low Risk', 'message': 'The duration of this single activity session appears reasonable.'}


# --- Button Callback Function for Key Generation ---
def update_key_and_rerun():
    """Generates a new key and forces the text input value to update."""
    new_key = generate_key()
    # 1. Update the application's core state variable
    st.session_state.fernet_key = new_key
    # 2. CRITICAL FIX: Update the text input widget's state variable directly
    st.session_state.key_verifier = new_key 
    st.rerun()

# --- Streamlit Page Setup ---
st.set_page_config(page_title="HAR Model", page_icon="🏃", layout="wide")
st.title("🏃 Human Activity Recognition (HAR) Model")
st.write("Upload your data below for training, and then use the prediction tool and analysis charts.")
st.markdown("---")

# Initialize session state for the key
if 'fernet_key' not in st.session_state:
    st.session_state.fernet_key = ""

# =========================================================================
# === SIDEBAR CONTENT: KEY MANAGEMENT AND ENCRYPTION/DECRYPTION ===
# =========================================================================

# --- 1. Key Management Section (Sidebar) ---
st.sidebar.header("1. 🔑 Key Management")
st.sidebar.markdown("Generate or paste your **Fernet Key** here. This key is required for encrypting your data (Step 2) and decrypting it (Step 3).")

# The text input is initialized with the current fernet_key state.
# Its value is implicitly stored under st.session_state.key_verifier
key_input_bar = st.sidebar.text_input(
    "Enter/Verify your Fernet Key:", 
    value=st.session_state.fernet_key,
    key="key_verifier",
    help="Paste your saved key here. Typing/pasting automatically sets the key for the current session."
)

# After the text input has been rendered and its state potentially updated, 
# we synchronize the core application state (fernet_key) with the widget's state (key_verifier)
st.session_state.fernet_key = st.session_state.key_verifier

# Use the callback function to handle button click and state update
if st.sidebar.button("Generate New Key", key="gen_key_button", on_click=update_key_and_rerun):
    pass # Logic is handled by the on_click callback

if st.session_state.fernet_key:
    st.sidebar.info(f"**Current Active Key:** `{st.session_state.fernet_key}`")
    st.sidebar.warning("⚠️ **Reminder:** Copy and save this key securely!")

st.sidebar.markdown("---")

# --- 2. Data Encryption Utility (Sidebar) ---
st.sidebar.header("2. 🔒 Encrypt Data Utility")
st.sidebar.markdown("Convert your original CSV into a secure, encrypted file for later use.")

if not st.session_state.fernet_key:
    st.sidebar.warning("Please enter or generate a key in Step 1 to enable encryption.")
else:
    original_file = st.sidebar.file_uploader("Upload Original (Unencrypted) CSV", type=["csv"], key="encrypt_uploader")

    if original_file:
        original_data_bytes = original_file.getvalue()
        
        # Encrypt the data immediately
        encrypted_data = encrypt_data(st.session_state.fernet_key, original_data_bytes)
        
        # Streamlit's download button provides the file directly
        st.sidebar.download_button(
            label="Encrypt & Download Secure File",
            data=encrypted_data,
            file_name="encrypted_strava_data.csv",
            mime="application/octet-stream",
            key="encrypt_download_button_direct",
            help="Click to encrypt the uploaded file and download the secure version."
        )
        st.sidebar.success("Encrypted file ready for download!")


st.sidebar.markdown("---")

# --- 3. Encrypted File Decryption Upload (Sidebar) ---
st.sidebar.header("3. 🔓 Load Encrypted Data")
st.sidebar.markdown("Upload the **encrypted** file and use your key for decryption.")

encrypted_file = st.sidebar.file_uploader(
    "Upload Encrypted Strava Dataset (.csv)", 
    type=["csv"], 
    key="decrypt_uploader",
    help="Upload the file secured in Step 2. The key from Step 1 is required for decryption."
)

st.sidebar.markdown("---")


# =========================================================================
# === MAIN TAB CONTENT: DATA LOAD, TRAINING, PREDICTION, AND EVALUATION ===
# =========================================================================

# --- 4. Data Loading Workflow (Main Tab) ---
st.header("4. Load Data & Start Analysis")
st.markdown("Upload your **original, unencrypted CSV** here for immediate analysis, or use the sidebar to load an encrypted file.")

df = None
analysis_key = st.session_state.fernet_key

# --- Plain File Upload (Main Tab) ---
plain_file = st.file_uploader(
    "Upload Plain (Unencrypted) Strava CSV", 
    type=["csv"], 
    key="plain_uploader",
    help="Upload the original file directly for immediate analysis."
)

file_to_process = None
file_type = None

# Determine which file to process (encrypted takes precedence if both are uploaded)
if encrypted_file:
    file_to_process = encrypted_file
    file_type = 'encrypted'
elif plain_file:
    file_to_process = plain_file
    file_type = 'plain'

# --- Data Loading and Initial Preparation ---
if file_to_process:
    
    # Decryption/Loading Logic
    if file_type == 'encrypted':
        if not analysis_key:
            st.error("Cannot decrypt. Please ensure your Fernet key is active in the sidebar's Step 1.")
        else:
            encrypted_bytes = file_to_process.getvalue()
            decrypted_bytes = decrypt_data(analysis_key, encrypted_bytes)

            if decrypted_bytes is not None:
                try:
                    decrypted_string = decrypted_bytes.decode('utf-8')
                    df = pd.read_csv(StringIO(decrypted_string))
                    st.success("✅ Encrypted Dataset Decrypted and Loaded Successfully! Analyzing...")
                except Exception as e:
                    st.error(f"Error processing decrypted file. Key might be wrong or the file is corrupted. Error: {e}")
                    df = None
            else:
                st.error("Decryption failed. Please ensure the key in the sidebar is correct for this encrypted file.")

    elif file_type == 'plain':
        try:
            plain_bytes = file_to_process.getvalue()
            plain_string = plain_bytes.decode('utf-8')
            df = pd.read_csv(StringIO(plain_string))
            st.success("✅ Plain Dataset Loaded Successfully! Analyzing...")
        except Exception as e:
            st.error(f"Error reading plain CSV file: {e}")
            df = None

    # --- Data Cleaning and Feature Engineering (Post-Load) ---
    if df is not None and not df.empty:
        try:
            # 1. Cleaning
            df = df.loc[:, ~df.columns.duplicated()]
            required_cols = ['distance', 'average_speed', 'max_speed', 'type']
            df = df.dropna(subset=required_cols)
            
            # 2. Unit Standardization (Kaggle Alignment: m/s -> m/min if needed)
            df['average_speed'] = df['average_speed'].apply(standardize_speed)
            df['max_speed'] = df['max_speed'].apply(standardize_speed)

            # 3. Calculate moving_time (in minutes) from distance (m) and average_speed (m/min)
            # Time (min) = (Distance (m) / Speed (m/min))
            df['moving_time'] = df.apply(
                lambda row: (row['distance'] / row['average_speed']) if row['average_speed'] > 0 else 0,
                axis=1
            )
            
            # 4. Feature Creation (Ratio is unit agnostic as long as units are consistent)
            df['intensity_ratio'] = df['average_speed'] / df['max_speed']
            df['intensity_ratio_squared'] = df['intensity_ratio'] ** 2 

            # 5. Target Labeling
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


# --- Model Training and Prediction Section ---
if df is not None and not df.empty:
    
    # --- Unified Model Training (Single Classifier) ---
    st.header("5. Model Training Status")
    
    # KAGGLE ALIGNMENT: Include distance and moving_time as features
    feature_columns = [
        'distance', 
        'moving_time',
        'average_speed', 
        'max_speed', 
        'intensity_ratio', 
        'intensity_ratio_squared'
    ]

    if all(col in df.columns for col in feature_columns):
        X = df[feature_columns]
        y = df['activity_label']
        
        # Use more robust parameters matching the notebook
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y) 
        
        unified_classifier = RandomForestClassifier(
            n_estimators=1000, 
            random_state=42, 
            max_depth=15, 
            class_weight='balanced'
        )
        unified_classifier.fit(X_train, y_train)

        st.success("✅ Random Forest Classifier (Unified 3-Class Model) trained successfully on your data (using distance, time, and speed features).")
        
        # --- Prepare for Prediction and Evaluation ---
        st.markdown("---")
        st.header("6. Real-time Prediction & Overtraining Check")

        # --- Integrated Prediction Function (Using the UNIFIED Model) ---
        def predict_activity_unified(distance, moving_time, avg_speed_input, max_speed_input):
            
            # CRITICAL: Apply the same standardization as the training data
            avg_speed = standardize_speed(avg_speed_input)
            max_speed = standardize_speed(max_speed_input)

            if max_speed == 0:
                 max_speed = 0.1 # Prevent ZeroDivisionError for ratio calc

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

        # --- User Input & Prediction ---
        col_dist, col_time, col_weight = st.columns(3)
        col_avg, col_max = st.columns(2)

        with col_dist:
            # Distance in meters
            distance = st.number_input("Distance (m)", min_value=0.0, value=25000.0, step=1000.0, help="Total distance covered in meters (SIM_DISTANCE).")
        with col_time:
            # Moving Time in minutes (SIM_MOVING_TIME)
            moving_time = st.number_input("Moving Time (min)", min_value=1.0, value=60.0, step=1.0, help="Total time spent moving in minutes (SIM_MOVING_TIME).")
        with col_weight:
            # Weight in kg for calorie calculation (SIM_USER_WEIGHT)
            user_weight = st.number_input("Your Weight (kg)", min_value=10.0, value=75.0, step=5.0, help="Your body weight for calorie calculation (SIM_USER_WEIGHT).")
            
        with col_avg:
            # Average speed in m/s (SIM_AVG_SPEED)
            avg_speed = st.number_input("Average Speed (m/s)", min_value=0.1, value=6.94, step=0.1, help="The average speed for the activity, in meters per second (SIM_AVG_SPEED).")
        with col_max:
            # Max speed in m/s (SIM_MAX_SPEED)
            max_speed = st.number_input("Max Speed (m/s)", min_value=0.1, value=9.0, step=0.1, help="The peak speed achieved, in meters per second (SIM_MAX_SPEED).")

        if st.button("🚀 Predict Activity & Calories"):
            
            if avg_speed > max_speed:
                st.error("Average speed cannot exceed Max speed.")
            elif avg_speed <= 0 or max_speed <= 0:
                 st.error("Speed values must be greater than zero.")
            else:
                # Prediction uses the full set of 4 user inputs
                predicted_activity = predict_activity_unified(distance, moving_time, avg_speed, max_speed)
                predicted_calories = estimate_calories(predicted_activity, moving_time, user_weight)
                
                overtraining_result = check_overtraining_risk(predicted_activity, moving_time)

                st.success(f"🏃 **Predicted Activity:** {predicted_activity}")
                st.info(f"🔥 **Estimated Calories Burned:** {predicted_calories:.2f} kcal")

                st.subheader("Health Guidance")
                if overtraining_result['risk'] != 'Low Risk':
                    st.warning(f"🚨 **Overtraining Risk Check:** {overtraining_result['risk']}")
                    st.markdown(f"**Guidance:** {overtraining_result['message']}")
                else:
                    st.info(f"🧘 **Overtraining Risk Check:** {overtraining_result['risk']}")
                    st.markdown(f"**Guidance:** {overtraining_result['message']}")


                # --- Evaluation Section ---
                st.markdown("---")
                st.header("7. Model Evaluation Results")
                
                y_pred_unified = unified_classifier.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred_unified)
                st.write(f"**Overall 3-Class Accuracy (Walk, Run, Ride):** {accuracy * 100:.2f}%")
                
                # --- Detailed Classification Metrics ---
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
                
                # --- Visualizations ---
                col_cm, col_imp = st.columns(2)
                
                with col_cm:
                    st.subheader("🔷 Confusion Matrix")
                    
                    cm = confusion_matrix(y_test, y_pred_unified, labels=classes)
                    fig, ax = plt.subplots(figsize=(6, 6))
                    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
                    disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
                    
                    ax.set_title("Unified HAR Classifier Results", fontsize=14)
                    ax.set_xlabel("Predicted Label", fontsize=12)
                    ax.set_ylabel("True Label", fontsize=12)

                    st.pyplot(fig)
                
                with col_imp:
                    st.subheader("📊 Feature Importance")
                    
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
