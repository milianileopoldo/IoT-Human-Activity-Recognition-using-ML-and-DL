# Imports
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix, classification_report
from cryptography.fernet import Fernet
from io import StringIO


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
        # Note: If the file was encrypted with a different key, this will raise an exception.
        f = Fernet(key.encode())
        return f.decrypt(encrypted_bytes)
    except Exception:
        # Catch decryption errors (wrong key, corrupted data)
        return None

# --- Calorie Estimation Function ---
def estimate_calories(activity_type, moving_time, weight):
    """Calculates estimated calories burned using standard METs values (METs * Wt * Time_Hours)."""
    time_in_hours = moving_time / 60.0
    
    if activity_type.lower() == 'walk': mets = 3.5 
    elif activity_type.lower() == 'run': mets = 9.0 
    elif activity_type.lower() == 'ride': mets = 6.0 
    else: mets = 5.0 
        
    return mets * weight * time_in_hours

# --- Overtraining Risk Check Function ---
def check_overtraining_risk(activity_type, moving_time):
    """
    Assesses a single activity session for high volume relative to general health guidelines.
    Overtraining Syndrome (OTS) depends on weekly load, but this flags potentially excessive single sessions.
    """
    moving_time_minutes = moving_time 

    if activity_type in ['Run', 'Ride']:
        # Flag vigorous activities over 60 minutes as high volume
        if moving_time_minutes > 60:
            return {
                'risk': 'High Volume / Vigorous Alert',
                'message': (
                    "This is a vigorous activity session lasting over an hour. "
                    "This volume is typically associated with targeted training plans. "
                    "Ensure this fits your weekly schedule and prioritize recovery days "
                    "to prevent potential overreaching."
                )
            }
    elif activity_type == 'Walk':
        # Flag moderate activities (walk) over 120 minutes as long duration
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


# --- Streamlit Page Setup ---
st.set_page_config(page_title="HAR Model", page_icon="🏃", layout="centered")
st.title("🔒 Private HAR Model – Decryption and Analysis")
st.write("Secure your Strava data using Fernet encryption. The key is required to unlock your dataset.")
st.markdown("---")

# Initialize session state for the key
if 'fernet_key' not in st.session_state:
    st.session_state.fernet_key = ""

# --- 1. Key Management Section ---
st.header("1. Key Management (Crucial Step)")
st.markdown("""
**How the key is handled:**
* You must generate a key and save it.
* This key is required to encrypt your original file (Step 2) and to decrypt the encrypted file for analysis (Step 3).
""")

col_key_input, col_key_gen = st.columns([3, 1])

with col_key_input:
    key_input = st.text_input(
        "Enter/Verify your Fernet Key:", 
        value=st.session_state.fernet_key,
        help="Paste your saved key here, or use the key generated below."
    )
    if key_input and key_input != st.session_state.fernet_key:
        st.session_state.fernet_key = key_input

with col_key_gen:
    if st.button("Generate New Key"):
        st.session_state.fernet_key = generate_key()
        st.success("New Key Generated!")

if st.session_state.fernet_key:
    st.info(f"**Current Key:** `{st.session_state.fernet_key}`")
    st.warning("⚠️ **ACTION REQUIRED:** Copy and save this key securely. If you lose it, your encrypted data will be lost!")

st.markdown("---")

# --- 2. Data Encryption Utility (One-Time) ---
st.header("2. Encrypt Your Original Data (Optional Utility)")

if not st.session_state.fernet_key:
    st.warning("Please generate or enter a key in Step 1 to enable encryption.")
else:
    original_file = st.file_uploader("Upload Original (Unencrypted) Strava CSV to Encrypt", type=["csv"], key="encrypt_uploader")

    if original_file:
        # Use a unique button key to prevent interaction conflicts
        if st.button("Encrypt & Download Secure File", key="encrypt_button"):
            try:
                original_data_bytes = original_file.getvalue()
                # Encrypt the raw bytes of the CSV file
                encrypted_data = encrypt_data(st.session_state.fernet_key, original_data_bytes)
                
                st.download_button(
                    label="Download Encrypted strava_data.csv",
                    data=encrypted_data,
                    file_name="encrypted_strava_data.csv",
                    mime="application/octet-stream"
                )
                st.success("File encrypted and ready for download. Use this new file in Step 3 if you wish to use the secure workflow later.")
            except Exception as e:
                st.error(f"Encryption failed: {e}")

st.markdown("---")

# --- 3. Analysis Workflow (Decryption or Direct Load) ---
st.header("3. Analysis Workflow (Encrypted or Plain Data)")
st.markdown("Upload either your **Encrypted** file (with the key) or your **Plain** CSV for immediate analysis.")

df = None
analysis_key = st.session_state.fernet_key

col_enc, col_plain = st.columns(2)

with col_enc:
    encrypted_file = st.file_uploader(
        "Upload Encrypted Strava Dataset (.csv)", 
        type=["csv"], 
        key="decrypt_uploader",
        help="Upload the file you secured in Step 2. Requires the key from Step 1."
    )
    
with col_plain:
    plain_file = st.file_uploader(
        "Upload Plain (Unencrypted) Strava CSV", 
        type=["csv"], 
        key="plain_uploader",
        help="Upload the original file directly for immediate analysis."
    )

file_to_process = None
file_type = None

# Determine which file to process
if encrypted_file:
    file_to_process = encrypted_file
    file_type = 'encrypted'
elif plain_file:
    file_to_process = plain_file
    file_type = 'plain'

# --- Data Loading and Initial Preparation ---
if file_to_process:
    if file_type == 'encrypted':
        if not analysis_key:
             st.error("Cannot decrypt. Please ensure your Fernet key is active in Step 1.")
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
                st.error("Decryption failed. Please ensure the key in Step 1 is correct for this encrypted file.")

    elif file_type == 'plain':
        try:
            plain_bytes = file_to_process.getvalue()
            plain_string = plain_bytes.decode('utf-8')
            df = pd.read_csv(StringIO(plain_string))
            st.success("✅ Plain Dataset Loaded Successfully! Analyzing...")
        except Exception as e:
            st.error(f"Error reading plain CSV file: {e}")
            df = None

# --- Start Model Training/Prediction ONLY if decryption/loading was successful ---
if df is not None and not df.empty:
    
    # --- Prepare and clean data ---
    df = df.loc[:, ~df.columns.duplicated()]
    required_cols = ['distance', 'moving_time', 'average_speed', 'max_speed', 'total_elevation_gain', 'type']
    df = df.dropna(subset=required_cols)

    # --- Feature Engineering & Target Labeling ---
    
    # 1. Feature Engineering
    df['intensity_ratio'] = df['average_speed'] / df['max_speed']
    df['intensity_ratio_squared'] = df['intensity_ratio'] ** 2 

    # 2. Target Labeling (based on simplified rules to force 3 classes)
    # The 'type' column in the original data provides the ground truth (Walk, Run, Ride).
    df['activity_label'] = df['type'].apply(lambda x: x if x in ['Walk', 'Run', 'Ride'] else 'Other')
    df = df[df['activity_label'] != 'Other'] # Keep only the three target classes

    if df['activity_label'].nunique() < 3:
        st.error("Dataset needs at least 'Walk', 'Run', and 'Ride' activities for training. Please upload a dataset containing all three types.")
        df = None 

if df is not None and not df.empty:
    
    # --- Unified Model Training (Single Classifier) ---
    # Features for the single 3-class model
    feature_columns = [
        'average_speed', 
        'max_speed', 
        'intensity_ratio', 
        'intensity_ratio_squared'
    ]

    X = df[feature_columns]
    y = df['activity_label']
    
    # Split data (stratify ensures balanced classes in test set)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y) 
    
    # Train classifier
    unified_classifier = RandomForestClassifier(
        n_estimators=500, 
        random_state=42, 
        max_depth=10, 
        class_weight='balanced'
    )
    unified_classifier.fit(X_train, y_train)

    # --- Prepare for Prediction and Evaluation ---
    
    st.markdown("---")
    st.header("4. Predict New Activity & Check Overtraining")

    # --- Integrated Prediction Function (Using the UNIFIED Model) ---
    def predict_activity_unified(avg_speed, max_speed):
        if max_speed == 0: 
            return 'Walk' 

        ratio = avg_speed / max_speed
        ratio_sq = ratio ** 2
        
        features = pd.DataFrame([{
            'average_speed': avg_speed, 
            'max_speed': max_speed, 
            'intensity_ratio': ratio, 
            'intensity_ratio_squared': ratio_sq
        }])
        
        return unified_classifier.predict(features)[0]

    # --- User Input & Prediction ---
    distance = st.number_input("Distance (m)", min_value=0.0, value=10000.0, step=1000.0)
    moving_time = st.number_input("Moving Time (min)", min_value=1.0, value=50.0, step=1.0)
    user_weight = st.number_input("Your Weight (kg)", min_value=10.0, value=70.0, step=5.0)
    avg_speed = st.number_input("Average Speed (m/min)", min_value=0.1, value=200.0, step=1.0)
    max_speed = st.number_input("Max Speed (m/min)", min_value=0.1, value=220.0, step=1.0)
    elevation_gain = st.number_input("Total Elevation Gain (m)", min_value=0.0, value=25.0, step=10.0)

    if st.button("🚀 Predict Activity & Calories"):
        predicted_activity = predict_activity_unified(avg_speed, max_speed)
        predicted_calories = estimate_calories(predicted_activity, moving_time, user_weight)
        
        # --- Overtraining Risk Check ---
        overtraining_result = check_overtraining_risk(predicted_activity, moving_time)

        st.success(f"🏃 **Predicted Activity:** {predicted_activity}")
        st.info(f"🔥 **Estimated Calories Burned:** {predicted_calories:.2f} kcal")

        # Display Overtraining Check Result
        if overtraining_result['risk'] != 'Low Risk':
            st.warning(f"🚨 **Overtraining Risk Check:** {overtraining_result['risk']}")
            st.markdown(f"**Guidance:** {overtraining_result['message']}")
        else:
            st.info(f"🧘 **Overtraining Risk Check:** {overtraining_result['risk']}")
            st.markdown(f"**Guidance:** {overtraining_result['message']}")


        # --- Evaluation Section ---
        st.header("5. Model Evaluation (Unified 3-Class Model)")
        
        # Evaluate the single unified model
        y_pred_unified = unified_classifier.predict(X_test)
        
        # Calculate overall accuracy and display as percentage
        accuracy = accuracy_score(y_test, y_pred_unified)
        st.write(f"**Overall 3-Class Accuracy (Walk, Run, Ride):** {accuracy * 100:.2f}%")
        
        # --- Detailed Classification Metrics (New Section) ---
        st.subheader("Detailed Classification Metrics")
        
        # Get report and convert to DataFrame for structured display
        classes = sorted(y_test.unique())
        report = classification_report(y_test, y_pred_unified, target_names=classes, output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report).transpose()
        
        # Convert metrics columns (precision, recall, f1-score) to percentage format
        metrics_to_format = ['precision', 'recall', 'f1-score']
        for col in metrics_to_format:
            if col in report_df.columns:
                # Multiply by 100 and format as a string with '%'
                report_df[col] = (report_df[col] * 100).apply(lambda x: f"{x:.2f}%" if pd.notna(x) else x)

        st.dataframe(report_df)
        
        st.markdown("---")
        
        # Single Confusion Matrix
        st.subheader("🔷 Single Confusion Matrix")
        
        cm = confusion_matrix(y_test, y_pred_unified, labels=classes)
        fig, ax = plt.subplots(figsize=(7, 7))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        # Use a larger font size for labels
        disp.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
        
        # Title and Axis labels
        ax.set_title("Unified HAR Classifier Results", fontsize=14)
        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_ylabel("True Label", fontsize=12)

        st.pyplot(fig)
        
        # --- Feature Importance Plot (New Plot) ---
        st.markdown("---")
        st.subheader("📊 Feature Importance")
        
        # Get feature importances from the unified classifier
        importances = unified_classifier.feature_importances_
        feature_names = X_train.columns
        
        # Create a DataFrame for easy sorting and plotting
        feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=True)

        fig_imp, ax_imp = plt.subplots(figsize=(8, 5))
        ax_imp.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='teal')
        ax_imp.set_xlabel("Feature Importance Score (Random Forest)", fontsize=12)
        ax_imp.set_title("Feature Importance in Unified HAR Model", fontsize=14)
        
        st.pyplot(fig_imp)
