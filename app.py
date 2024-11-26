import streamlit as st
import numpy as np
import tensorflow as tf
import librosa
from tensorflow.keras.models import load_model

# Load the model
model = load_model('audio_classification.keras')

# Define class labels
class_labels = [
    "Air Conditioner", "Car Horn", "Children Playing", "Dog Bark", "Drilling",
    "Engine Idling", "Gun Shot", "Jackhammer", "Siren", "Street Music"
]

confidence_threshold = 0.4

# App title
st.title("AUDIO CLASSIFICATION")
st.write("Upload an audio file to identify the sound.")

col1, col2 = st.columns(2)
with col1:
    for label in class_labels[:5]:  # First half of the list
        st.write(f"- {label}")
with col2:
    for label in class_labels[5:]:  # Second half of the list
        st.write(f"- {label}")

# File uploader
uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"])

def preprocess_audio(file):
    # Load and preprocess audio file
    y, sr = librosa.load(file, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc = np.mean(mfcc.T, axis=0)
    return np.expand_dims(mfcc, axis=0)

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    
    # Process the file and predict
    if st.button("Predict"):
        try:
            features = preprocess_audio(uploaded_file)
            prediction = model.predict(features)
            predicted_index = np.argmax(prediction, axis=1)[0]
            predicted_label = class_labels[predicted_index]
            confidence = np.max(prediction)

            # Display result
            st.write(f"**Predicted Sound**: {predicted_label}")
            st.write(f"**Confidence**: {confidence:.2f}")
        
            if confidence < confidence_threshold:
                st.write("⚠️ *The confidence is quite low, so it's less likely that this prediction is accurate.*")
        
        except Exception as e:
            st.error(f"Error in processing: {e}")
