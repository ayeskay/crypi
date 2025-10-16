import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- 1. Setup & Model Loading ---
DEVICE = torch.device('cpu')
MODEL_PATH = './final_model'   # Local model folder

print(f"Loading model from: {MODEL_PATH}")
try:
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model.to(DEVICE)
    model.eval()
    print("✅ Model loaded successfully and ready for inference.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise e

# --- 2. FastAPI Application ---
app = FastAPI(title="Crypi Security Prediction API")

# --- 2a. Add CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # allow all origins; change in production
    allow_credentials=True,
    allow_methods=["*"],          # allows POST, GET, OPTIONS, etc.
    allow_headers=["*"],          # allow all headers
)

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
        "probabilities": {
            "vulnerable": probs[0][0].item(),
            "secure": probs[0][1].item()
        }
    }

@app.post("/predict")
def predict(snippet: CodeSnippet):
    return predict_code_security(snippet.text)

# --- 3. Run the server automatically ---
if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Crypi API server on http://127.0.0.1:8000 ...")
    uvicorn.run(app, host="0.0.0.0", port=8000)