

import gradio as gr
import tensorflow as tf
import numpy as np
import joblib

# Load model
model = tf.keras.models.load_model("healthcare_chatbot_model.keras")

# Load saved objects
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

    if symptoms.strip() == "":
        return "Please enter symptoms.", "", ""

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

demo = gr.Interface(
    fn=chatbot,
    inputs=gr.Textbox(
        label="Enter Symptoms",
        placeholder="Example: itching, skin_rash, headache"
    ),
    outputs=[
        gr.Textbox(label="Predicted Disease"),
        gr.Textbox(label="Healthcare Advice"),
        gr.Textbox(label="Medical Information")
    ],
    title="🏥 AI-Based Healthcare Chatbot",
    description="Enter symptoms separated by commas."
)

demo.launch()
