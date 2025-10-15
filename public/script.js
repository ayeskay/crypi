document.addEventListener('DOMContentLoaded', () => {
    const codeInput = document.getElementById('codeInput');
    const checkButton = document.getElementById('checkButton');
    const resultContainer = document.getElementById('resultContainer');
    const resultContent = document.getElementById('resultContent');
    const loader = document.getElementById('loader');
    const predictionText = document.getElementById('predictionText');
    const confidenceScore = document.getElementById('confidenceScore');
    const secureBar = document.getElementById('secureBar');
    const vulnBar = document.getElementById('vulnBar');
    const secureLabel = document.getElementById('secureLabel');
    const vulnLabel = document.getElementById('vulnLabel');

    checkButton.addEventListener('click', async () => {
        const snippet = codeInput.value.trim();
        if (!snippet) { alert('Please enter a code snippet.'); return; }
        resultContainer.classList.remove('hidden');
        resultContent.classList.add('hidden');
        loader.classList.remove('hidden');
        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: snippet }),
            });
            if (!response.ok) throw new Error(`API Error: ${response.statusText}`);
            const data = await response.json();
            displayResults(data);
        } catch (error) {
            displayError(error.message);
        } finally {
            loader.classList.add('hidden');
            resultContent.classList.remove('hidden');
        }
    });

    function displayResults(data) {
        const { prediction, probabilities } = data;
        const secureProb = probabilities.secure * 100;
        const vulnProb = probabilities.vulnerable * 100;
        const confidence = Math.max(secureProb, vulnProb);
        predictionText.textContent = prediction;
        predictionText.className = prediction.toLowerCase();
        confidenceScore.textContent = confidence.toFixed(2);
        secureBar.style.width = `${secureProb}%`;
        vulnBar.style.width = `${vulnProb}%`;
        secureLabel.textContent = `Secure: ${secureProb.toFixed(2)}%`;
        vulnLabel.textContent = `Vulnerable: ${vulnProb.toFixed(2)}%`;
    }

    function displayError(errorMessage) {
        predictionText.textContent = 'Error';
        predictionText.className = 'vulnerable';
        confidenceScore.textContent = "N/A";
        secureLabel.textContent = `An error occurred: ${errorMessage}`;
        vulnLabel.textContent = "";
        secureBar.style.width = '0%';
        vulnBar.style.width = '0%';
    }
});
