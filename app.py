

from flask import Flask, request, render_template
import torch
import torch.nn as nn
import joblib
import numpy as np

# Define the model architecture
class SpamClassifier(nn.Module):
    def __init__(self, input_dim=1000):
        super(SpamClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)

# Load model and vectorizer
model = SpamClassifier()
model.load_state_dict(torch.load("spam_classifier.pth", map_location=torch.device('cpu')))
model.eval()

tfidf = joblib.load("tfidf_vectorizer.pkl")

# Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def predict():
    input_text = request.form['email']
    features = tfidf.transform([input_text]).toarray()
    with torch.no_grad():
        tensor_input = torch.tensor(features, dtype=torch.float32)
        output = model(tensor_input)
        _, predicted = torch.max(output, 1)
        label = "Spam" if predicted.item() == 1 else "Not Spam"
    return render_template('index.html', prediction=label)

if __name__ == '__main__':
    app.run(debug=True)
