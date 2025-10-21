# --- HAR MODEL: Human Activity Recognition App ---
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, confusion_matrix, ConfusionMatrixDisplay

# --- Streamlit Page Setup ---
st.set_page_config(page_title="HAR Model", page_icon="🏃", layout="centered")
st.title("🏃 HAR MODEL – Human Activity Recognition from Strava Data")
st.write("Upload your dataset and predict your activity type and estimated calories burned using Machine Learning.")

# --- File Upload ---
uploaded_file = st.file_uploader("📁 Upload your Strava Dataset (CSV Format)", type=["csv"])

if uploaded_file is not None:
    # --- Load Dataset ---
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset Loaded Successfully!")

    # --- Clean Columns ---
    df = df.loc[:, ~df.columns.duplicated()]
    df.rename(columns={
        'Activity Type': 'type',
        'Distance': 'distance',
        'Moving Time': 'moving_time',
        'Elevation Gain': 'total_elevation_gain',
        'Average Speed': 'average_speed',
        'Max Speed': 'max_speed'
    }, inplace=True)

    # --- Create Features ---
    df['distance_km'] = df['distance']
    df['moving_time_min'] = df['moving_time'] / 60
    df['average_speed_kmh'] = df['average_speed']
    df['max_speed_kmh'] = df['max_speed']
    df['intensity_ratio'] = df['average_speed_kmh'] / df['max_speed_kmh']
    df = df.dropna(subset=['distance_km', 'moving_time_min', 'average_speed_kmh', 'max_speed_kmh'])

    # --- Generate Activity Labels ---
    def classify_activity(row):
        score = (0.7 * row['average_speed_kmh']) + (30 * row['intensity_ratio'])
        if score < 9:
            return "Walk"
        elif score < 20:
            return "Run"
        else:
            return "Ride"
    df['activity_label'] = df.apply(classify_activity, axis=1)

    # --- Define Features ---
    feature_columns = [
        'distance_km',
        'moving_time_min',
        'total_elevation_gain',
        'average_speed_kmh',
        'max_speed_kmh',
        'intensity_ratio'
    ]
    X = df[feature_columns]
    y = df['activity_label']

    # --- Split Data ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # --- Train Classifier ---
    activity_model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        max_depth=10,
        class_weight='balanced'
    )
    activity_model.fit(X_train, y_train)

    # --- Train Calorie Regressor ---
    def estimate_calories(activity_type, distance_km, weight=70):
        if activity_type.lower() == 'walk':
            return 0.75 * weight * distance_km
        elif activity_type.lower() == 'run':
            return 1.05 * weight * distance_km
        elif activity_type.lower() == 'ride':
            return 0.30 * weight * distance_km
        else:
            return 0.50 * weight * distance_km

    df['estimated_calories'] = df.apply(lambda row: estimate_calories(row['activity_label'], row['distance_km']), axis=1)
    X_cal = df[feature_columns]
    y_cal = df['estimated_calories']
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_cal, y_cal, test_size=0.3, random_state=42)

    calorie_model = RandomForestRegressor(n_estimators=150, random_state=42)
    calorie_model.fit(Xc_train, yc_train)

    # --- User Input Section ---
    st.header("🧍 Predict Your Activity")
    distance_km = st.number_input("Distance (km)", min_value=0.0, value=3.0, step=0.1)
    moving_time_min = st.number_input("Moving Time (minutes)", min_value=1.0, value=25.0, step=1.0)
    elevation_gain = st.number_input("Total Elevation Gain (m)", min_value=0.0, value=20.0, step=1.0)
    avg_speed = st.number_input("Average Speed (km/h)", min_value=0.1, value=7.0, step=0.1)
    max_speed = st.number_input("Max Speed (km/h)", min_value=0.1, value=11.0, step=0.1)

    intensity_ratio = avg_speed / max_speed if max_speed > 0 else 0
    new_data = pd.DataFrame([{
        'distance_km': distance_km,
        'moving_time_min': moving_time_min,
        'total_elevation_gain': elevation_gain,
        'average_speed_kmh': avg_speed,
        'max_speed_kmh': max_speed,
        'intensity_ratio': intensity_ratio
    }])

    # --- Predict Button ---
    if st.button("🚀 Predict"):
        predicted_activity = activity_model.predict(new_data)[0]
        predicted_calories = calorie_model.predict(new_data)[0]

        st.success(f"🏃 **Predicted Activity:** {predicted_activity}")
        st.info(f"🔥 **Estimated Calories Burned:** {predicted_calories:.2f} kcal")

        # --- Evaluation Metrics ---
        st.header("📈 Model Evaluation Results")

        # Accuracy + Classification Report
        y_pred = activity_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.write(f"**Accuracy:** {round(acc, 3)}")
        st.text(classification_report(y_test, y_pred, zero_division=1))

        # Confusion Matrix
        st.subheader("🔷 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred, labels=activity_model.classes_)
        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=activity_model.classes_)
        disp.plot(ax=ax, cmap="Blues", values_format="d")
        st.pyplot(fig)

        # Feature Importance
        st.subheader("📊 Feature Importance")
        importances = activity_model.feature_importances_
        features = pd.Series(importances, index=feature_columns).sort_values(ascending=True)
        fig2, ax2 = plt.subplots()
        ax2.barh(features.index, features.values, color='teal')
        ax2.set_title("Feature Importance in Activity Prediction")
        st.pyplot(fig2)

        # Calorie Model Accuracy
        calorie_error = mean_absolute_error(yc_test, calorie_model.predict(Xc_test))
        st.write(f"🔥 **Calorie Model MAE:** {round(calorie_error, 2)} kcal")
else:
    st.warning("👆 Please upload a valid Strava CSV file to continue.")

