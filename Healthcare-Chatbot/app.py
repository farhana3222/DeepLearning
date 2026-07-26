import streamlit as st
import tensorflow as tf
import numpy as np
import joblib

# Load model and files
model = tf.keras.models.load_model("healthcare_chatbot_model.keras")

symptom_list = joblib.load("symptom_list.pkl")
encoder = joblib.load("label_encoder.pkl")
dialogue_dict = joblib.load("dialogue_dict.pkl")
knowledge_base = joblib.load("knowledge_base.pkl")


def create_symptom_vector(user_input):
    vector = np.zeros(len(symptom_list))

    symptoms = [s.strip().lower() for s in user_input.split(",")]

    for symptom in symptoms:
        if symptom in symptom_list:
            idx = symptom_list.index(symptom)
            vector[idx] = 1

    return vector.reshape(1, -1)


def chatbot(symptoms):

    input_vector = create_symptom_vector(symptoms)

    prediction = model.predict(input_vector, verbose=0)

    predicted_label = np.argmax(prediction)

    disease = encoder.inverse_transform([predicted_label])[0]

    advice = dialogue_dict.get(
        disease.lower(),
        "Please consult a doctor."
    )

    medical_info = knowledge_base.get(
        disease.lower(),
        "No additional medical information available."
    )

    return disease, advice, medical_info


# ---------------- UI ----------------

st.set_page_config(
    page_title="AI Healthcare Chatbot",
    page_icon="🏥"
)

st.title("🏥 AI-Based Healthcare Chatbot")

st.write(
    "Enter symptoms separated by commas to predict the disease."
)

symptoms = st.text_input(
    "Enter Symptoms",
    placeholder="Example: itching, skin_rash, headache"
)

if st.button("Predict Disease"):

    if symptoms.strip() == "":
        st.warning("Please enter symptoms.")

    else:

        disease, advice, medical_info = chatbot(symptoms)

        st.success(f"Predicted Disease: {disease}")

        st.subheader("Healthcare Advice")
        st.write(advice)

        st.subheader("Medical Information")
        st.write(medical_info)
    
