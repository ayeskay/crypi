# Crypi: An AI-Powered Code Security Scanner

Crypi is a proof-of-concept web application designed to demonstrate the use of a fine-tuned Transformer model for static code analysis. The application allows users to submit Java code snippets and receive real-time predictions regarding their security posture, classifying them as either "Secure" or "Vulnerable".

## Project Overview

Crypi applies Natural Language Processing (NLP) techniques to the domain of software security. By treating source code as a structured text format, a pre-trained language model (CodeBERT) has been fine-tuned to recognize patterns and syntactic structures that are statistically associated with security vulnerabilities.

The model classifies Java code snippets as either "Secure" or "Vulnerable" based on common vulnerability patterns such as:

- Hardcoded secrets (e.g., passwords, API keys)
- Weak or outdated cryptographic algorithms (e.g., MD5, SHA-1)
- Insecure generation of random numbers for cryptographic purposes

This application serves as a demonstration of how machine learning can be applied to code security.

## Architectural Components

The system consists of two main components:

### 1. User Interface (Frontend)
The web interface is built using [Streamlit](https://streamlit.io/), providing an interactive platform for users to input code and receive analysis results.

### 2. Machine Learning Model (Backend Logic)
The core of the application is a **CodeBERT** model, specifically a fine-tuned version of **RobertaForSequenceClassification**. This model processes the input Java code and outputs a binary classification ("Secure" or "Vulnerable") along with a confidence score.

The model is hosted on the [Hugging Face Hub](https://huggingface.co/) for efficient deployment, separating the large model files from the source code.

## Model Details

The model has been trained to detect common vulnerabilities, including:

- **Hardcoded secrets:** Identifies the presence of hardcoded API keys, passwords, etc.
- **Weak cryptography:** Flags the use of outdated cryptographic algorithms like MD5 or SHA-1.
- **Insecure randomness:** Detects poor practices in random number generation for cryptographic purposes.

## Installation and Execution

Follow the steps below to run the application locally:

### Prerequisites

- **Python 3.8+**: A modern version of Python is required.
- **Git & Git LFS**: Git is required for cloning repositories, and Git LFS is needed to download the model files.

#### Installation on Debian/Ubuntu:

```bash
sudo apt-get update
sudo apt-get install git-lfs
````

#### Installation on macOS (via Homebrew):

```bash
brew install git-lfs
```

After installing Git LFS, run the following command to initialize it:

```bash
git lfs install
```

### Step 1: Clone the Application Repository

Clone the primary application repository from GitHub:

```bash
git clone https://github.com/ayeskay/crypi.git
cd crypi
```

### Step 2: Download the Fine-Tuned Model

Clone the model repository from Hugging Face Hub into a directory called `final_model`:

```bash
git clone https://huggingface.co/ayeskay/crypi final_model
```

Your local project structure should look like this:

```text
crypi/
├── streamlit_app.py
├── requirements.txt
└── final_model/        <-- Contains the model files
    ├── config.json
    ├── model.safetensors
    └── ... (tokenizer files, etc.)
```

### Step 3: Install Required Dependencies

Install the necessary Python packages using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### Step 4: Launch the Streamlit Application

Start the application by running the following command:

```bash
streamlit run streamlit_app.py
```

The application will open in your default web browser.

## Usage

1. Open the Crypi web interface.
2. Paste a Java code snippet into the provided text area.
3. Click the "Check Security" button.
4. The model will analyze the code and display the security classification ("Secure" or "Vulnerable") along with a confidence score and probability distribution.

## Disclaimer

This application is intended for educational and demonstration purposes only. It should **not** be used as a comprehensive security auditing tool or a substitute for professional code reviews, penetration testing, or commercial-grade Static Application Security Testing (SAST) solutions.

The model's predictions are based on statistical patterns and are subject to false positives and false negatives. No guarantees are made regarding the accuracy of its predictions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

* **Streamlit** for building the interactive web framework.
* **Hugging Face** for providing the pre-trained models and hosting platform.
* **CodeBERT** for enabling the application of NLP techniques to code security.
