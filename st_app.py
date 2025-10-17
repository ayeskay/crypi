import streamlit as st
import torch
from transformers import AutoTokenizer, RobertaForSequenceClassification # Use the specific class based on config
from scipy.special import softmax
import os
import numpy as np # Import numpy for probability processing

# --- Page Configuration (MUST BE FIRST Streamlit command) ---
st.set_page_config(page_title="Crypi Security Scanner", layout="centered")

# --- Configuration ---
LOCAL_MODEL_PATH = './final_model' # Path to the local directory
DEVICE = torch.device('cpu') # Assume CPU for Streamlit deployment unless specifically configured otherwise

# --- Load Model and Tokenizer (Cached for performance) ---
@st.cache_resource # Use Streamlit's caching for resources like models
def load_local_model_and_tokenizer():
    """Loads the tokenizer and model from the specified local directory."""
    if not os.path.isdir(LOCAL_MODEL_PATH):
        st.error(f"❌ Error: Local model directory not found at '{LOCAL_MODEL_PATH}'.")
        st.error("Please ensure the 'final_model' directory is in the same folder as this script.")
        st.stop() # Stop the app if the directory is missing

    try:
        print(f"Attempting to load tokenizer from local path: {LOCAL_MODEL_PATH}")
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)

        print(f"Attempting to load model from local path: {LOCAL_MODEL_PATH}")
        # Explicitly load the model type as RobertaForSequenceClassification
        model = RobertaForSequenceClassification.from_pretrained(LOCAL_MODEL_PATH)

        model.to(DEVICE)
        model.eval() # Set model to evaluation mode
        print("Model and tokenizer loaded successfully from local path.")
        return tokenizer, model
    except OSError as e:
        st.error(f"❌ Error loading model/tokenizer from '{LOCAL_MODEL_PATH}': {e}")
        st.error("Please ensure the directory contains all necessary files (config.json, model weights like model.safetensors or pytorch_model.bin, tokenizer files).")
        st.stop()
    except Exception as e:
        st.error(f"❌ An unexpected error occurred during model loading: {e}")
        st.stop()

# Load the resources; errors are handled within the function by st.stop()
tokenizer, model = load_local_model_and_tokenizer()

# --- Prediction Function ---
def predict_code_security(code_snippet):
    """Tokenizes snippet, runs inference using the loaded model, and returns prediction details."""
    if not code_snippet or not isinstance(code_snippet, str):
        st.warning("Input snippet is empty or invalid.")
        return None

    try:
        # Tokenize the input
        inputs = tokenizer(
            code_snippet,
            return_tensors='pt',
            truncation=True,
            max_length=512, # Ensure this matches training max length
            padding=True
        ).to(DEVICE)

        # Perform prediction
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            # Apply softmax to get probabilities
            probabilities_np = softmax(logits.cpu().numpy(), axis=1)[0]
            # Get the index of the highest probability
            prediction_idx = int(np.argmax(probabilities_np))

        # Map prediction index to label name (0=Vulnerable, 1=Secure)
        prediction_label = "SECURE" if prediction_idx == 1 else "VULNERABLE"
        # Get the confidence score for the predicted class
        confidence_score = probabilities_np[prediction_idx] * 100

        return {
            "prediction": prediction_label,
            "confidence": confidence_score,
            # Return individual probabilities
            "probabilities": {"vulnerable": float(probabilities_np[0]), "secure": float(probabilities_np[1])}
        }
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        return None

# --- Streamlit User Interface ---
st.title("Crypi - Local Code Security Scanner 🤖")
st.markdown("Enter a Java code snippet below to analyze it using the locally loaded model.")

# Text area for code input
code_input = st.text_area(
    "Code Snippet",
    height=250,
    placeholder="Paste your Java code snippet here...",
    key="code_input_area" # Assign a key for stability
    )

# Button to trigger analysis
analyze_button = st.button("Check Security", type="primary")

if analyze_button:
    if code_input:
        # Show spinner during processing
        with st.spinner("🕵️‍♀️ Analyzing code using the local model..."):
            analysis_result = predict_code_security(code_input)

        # Display results if analysis was successful
        if analysis_result:
            prediction = analysis_result["prediction"]
            confidence = analysis_result["confidence"]
            probs = analysis_result["probabilities"]
            vuln_prob = probs["vulnerable"] * 100
            sec_prob = probs["secure"] * 100

            st.subheader("Analysis Result")

            # Use success/error boxes based on prediction
            if prediction == "SECURE":
                st.success(f"Prediction: **{prediction}**")
            else:
                st.error(f"Prediction: **{prediction}**")

            # Display confidence
            st.metric(label="Confidence Score", value=f"{confidence:.2f}%")

            # Display probabilities clearly
            st.write("**Probability Distribution:**")
            col1, col2 = st.columns(2)
            with col1:
                # Ensure progress value is within [0, 1]
                progress_sec = min(1.0, max(0.0, sec_prob / 100.0))
                st.progress(progress_sec)
                st.markdown(f"🟢 Secure: `{sec_prob:.2f}%`")
            with col2:
                progress_vuln = min(1.0, max(0.0, vuln_prob / 100.0))
                st.progress(progress_vuln)
                st.markdown(f"🔴 Vulnerable: `{vuln_prob:.2f}%`")

        else:
            # If prediction failed after button click
            if code_input: # Check again if input was provided
                 st.warning("Could not complete the analysis. Please check for previous error messages or try modifying the input.")

    else:
        # If button clicked with no input
        st.warning("⚠️ Please enter a code snippet before clicking 'Check Security'.")

# Footer section
st.markdown("---")
st.caption("Powered by Transformers and Streamlit. Using local model from './final_model'.")
