import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# Step 1: Load and Preprocess Data
# Assume CSV file from Strava: 'strava_data.csv' with columns: timestamp, activity_type, heart_rate, pace, elevation, fatigue_level
data = pd.read_csv('strava_data.csv')  # Replace with your file path

# Encode HAR labels (e.g., 'running' -> 0, 'walking' -> 1, 'cycling' -> 2)
label_encoder = LabelEncoder()
data['activity_encoded'] = label_encoder.fit_transform(data['activity_type'])
num_classes = len(label_encoder.classes_)

# Select features for LSTM (time-series: heart_rate, pace, elevation)
features = ['heart_rate', 'pace', 'elevation']
X = data[features].values
y_har = to_categorical(data['activity_encoded'], num_classes=num_classes)  # One-hot for HAR
y_fatigue = data['fatigue_level'].values  # Binary: 0 or 1 (threshold at 0.5 if regression)

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Create sequences (e.g., 30 time steps per sample for LSTM)
sequence_length = 30  # Adjust based on your data (e.g., 30 seconds of data points)
X_seq, y_har_seq, y_fatigue_seq = [], [], []
for i in range(len(X) - sequence_length):
    X_seq.append(X[i:i+sequence_length])
    y_har_seq.append(y_har[i+sequence_length])
    y_fatigue_seq.append(y_fatigue[i+sequence_length])

X_seq = np.array(X_seq)
y_har_seq = np.array(y_har_seq)
y_fatigue_seq = np.array(y_fatigue_seq)

# Split into train/test
X_train, X_test, y_har_train, y_har_test, y_fatigue_train, y_fatigue_test = train_test_split(
    X_seq, y_har_seq, y_fatigue_seq, test_size=0.2, random_state=42
)

# Step 2: Build the LSTM Model with Multi-Output
input_layer = Input(shape=(sequence_length, len(features)))
lstm_layer = LSTM(64, return_sequences=False)(input_layer)  # 64 units; adjust as needed
dropout_layer = Dropout(0.2)(lstm_layer)

# HAR Output (multi-class)
har_output = Dense(num_classes, activation='softmax', name='har_output')(dropout_layer)

# Fatigue Output (binary classification; use 'sigmoid' for binary, 'linear' for regression)
fatigue_output = Dense(1, activation='sigmoid', name='fatigue_output')(dropout_layer)

model = Model(inputs=input_layer, outputs=[har_output, fatigue_output])

# Compile with losses for each output
model.compile(
    optimizer='adam',
    loss={'har_output': 'categorical_crossentropy', 'fatigue_output': 'binary_crossentropy'},
    metrics={'har_output': 'accuracy', 'fatigue_output': 'accuracy'}
)

# Step 3: Train the Model
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(
    X_train, {'har_output': y_har_train, 'fatigue_output': y_fatigue_train},
    validation_data=(X_test, {'har_output': y_har_test, 'fatigue_output': y_fatigue_test}),
    epochs=50, batch_size=32, callbacks=[early_stop]
)

# Step 4: Evaluate and Predict
loss, har_loss, fatigue_loss, har_acc, fatigue_acc = model.evaluate(
    X_test, {'har_output': y_har_test, 'fatigue_output': y_fatigue_test}
)
print(f"HAR Accuracy: {har_acc:.2f}, Fatigue Accuracy: {fatigue_acc:.2f}")

# Example Prediction on new data (simulate a sequence)
new_sequence = X_test[0:1]  # Take one test sample
har_pred, fatigue_pred = model.predict(new_sequence)
predicted_activity = label_encoder.inverse_transform([np.argmax(har_pred)])
fatigue_status = "Fatigued" if fatigue_pred[0][0] > 0.5 else "Not Fatigued"
print(f"Predicted Activity: {predicted_activity[0]}, Fatigue Status: {fatigue_status}")