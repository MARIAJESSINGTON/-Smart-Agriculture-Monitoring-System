import os
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# predict.py lives in src/
# models/ lives at the root (one level up from src/)
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ✅ Load models
crop_model = load_model(os.path.join(MODELS_DIR, "crop_dl_model.h5"))
soil_model = load_model(os.path.join(MODELS_DIR, "soil_dl_model.h5"))

# ✅ Load scalers & encoders
crop_scaler  = joblib.load(os.path.join(MODELS_DIR, "crop_scaler.pkl"))
crop_encoder = joblib.load(os.path.join(MODELS_DIR, "crop_encoder.pkl"))
soil_scaler  = joblib.load(os.path.join(MODELS_DIR, "soil_scaler.pkl"))
soil_encoder = joblib.load(os.path.join(MODELS_DIR, "soil_encoder.pkl"))


# 🌾 Crop Prediction
def predict_crop(data):
    data = np.array(data).reshape(1, -1)
    data = crop_scaler.transform(data)
    pred = crop_model.predict(data)
    result = crop_encoder.inverse_transform([np.argmax(pred)])
    return result[0]


# 💧 Irrigation Prediction
def predict_irrigation(data):
    data = np.array(data).reshape(1, -1)
    data = soil_scaler.transform(data)
    pred = soil_model.predict(data)
    result = soil_encoder.inverse_transform([np.argmax(pred)])
    return result[0]