import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- 1. Setup & Model Loading ---
DEVICE = torch.device('cpu')
# !!! IMPORTANT: This MUST match your Hugging Face username and repo name !!!
MODEL_REPO = 'ayeskay/crypi-code-security-model' 

# Load the trained model and tokenizer FROM THE HUB
# Vercel will download this automatically when the server starts.
model = AutoModelForSequenceClassification.from_pretrained(MODEL_REPO)
tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
model.to(DEVICE)
model.eval()

# --- 2. FastAPI Application ---
app = FastAPI(title="Crypi Security Prediction API")

class CodeSnippet(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"status": "Crypi API is running"}

def predict_code_security(code_snippet: str):
    inputs = tokenizer(
        code_snippet, return_tensors='pt', truncation=True, max_length=512, padding=True
    ).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        prediction = torch.argmax(probs, dim=-1).item()
    result = "SECURE" if prediction == 1 else "VULNERABLE"
    return {
        "prediction": result,
        "probabilities": {"vulnerable": probs[0][0].item(), "secure": probs[0][1].item()}
    }

@app.post("/predict")
def predict(snippet: CodeSnippet):
    return predict_code_security(snippet.text)
