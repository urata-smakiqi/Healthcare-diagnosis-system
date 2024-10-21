import pandas as pd

# Sample dataset of patient symptoms and medical history
data = {
    "Patient ID": [1, 2, 3],
    "Symptoms": ["fever, cough, fatigue", "chest pain, shortness of breath", "headache, nausea, dizziness"],
    "Medical History": ["asthma, hypertension", "smoker, heart disease", "migraine, no major illnesses"]
}

df = pd.DataFrame(data)
print(df)

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the FLAN-T5 large model from Hugging Face
model_name = "google/flan-t5-large"  # You can use other models as well
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

print("Model and tokenizer loaded successfully!")

def create_prompt(symptoms, medical_history):
    prompt = (f"Given the symptoms: {symptoms} and medical history: {medical_history}, "
              f"what are the possible diagnoses or suggested tests?")
    return prompt

# Example prompt for patient diagnosis support
prompt = create_prompt("fever, cough, fatigue", "asthma, hypertension")
print(prompt)

def generate_diagnosis(symptoms, medical_history):
    prompt = create_prompt(symptoms, medical_history)
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Generate output (possible diagnoses)
    outputs = model.generate(**inputs, max_length=100, num_beams=5, early_stopping=True)
    
    # Decode the output back to human-readable text
    diagnosis = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return diagnosis

# Test the model
symptoms = "fever, cough, fatigue"
medical_history = "asthma, hypertension"
diagnosis = generate_diagnosis(symptoms, medical_history)
print("Suggested diagnosis or tests:", diagnosis)

import streamlit as st

st.title("Healthcare Diagnosis Support System")

# Input fields
symptoms = st.text_input("Enter symptoms (comma-separated):")
medical_history = st.text_input("Enter medical history (comma-separated):")

if st.button("Get Diagnosis"):
    diagnosis = generate_diagnosis(symptoms, medical_history)
    st.write(f"Suggested diagnosis or tests: {diagnosis}")
