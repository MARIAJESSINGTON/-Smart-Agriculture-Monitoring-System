import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load dataset
df = pd.read_csv("../data/soil_data.csv")

# Clean column names
df.columns = df.columns.str.strip()

print("Columns:", df.columns)

# ✅ Create irrigation labels
def irrigation_label(row):
    if row["Soil_Moisture"] < 40:
        return "High"
    elif row["Soil_Moisture"] < 80:
        return "Medium"
    else:
        return "Low"

df["irrigation"] = df.apply(irrigation_label, axis=1)

# ✅ Features (USE EXACT NAMES)
X = df[["Soil_Moisture", "Soil_Temp", "Humidity"]]
y = df["irrigation"]

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)

# Scale
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Deep Learning Model
model = Sequential([
    Dense(32, activation='relu', input_shape=(3,)),
    Dense(16, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X, y, epochs=50)

# Save
model.save("../models/soil_dl_model.h5")
joblib.dump(scaler, "../models/soil_scaler.pkl")
joblib.dump(le, "../models/soil_encoder.pkl")

print("✅ Soil model trained successfully!")