from flask import Flask, request, render_template, jsonify
import torch
import torch.nn as nn
import numpy as np
from werkzeug.utils import secure_filename
import os

# Flask app setup
app = Flask(__name__)

# Load the model structure
class ImprovedIrisClassifier(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, dropout_rate=0.3):
        super(ImprovedIrisClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return self.softmax(x)

# Load the saved model
model = ImprovedIrisClassifier(input_size=4, hidden_size1=32, hidden_size2=16, output_size=3, dropout_rate=0.3)
model.load_state_dict(torch.load(r"C:\Users\Yashuyashash\iris flower classification\iris_model.pth"))
model.eval()

# Prediction function
def predict_flower(features):
    with torch.no_grad():
        inputs = torch.tensor([features], dtype=torch.float32)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

# Error handling
@app.errorhandler(Exception)
def handle_exception(e):
    return jsonify({"error": str(e)}), 500

# Routes
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # Extract input features from the form
            features = [
                float(request.form["sepal_length"]),
                float(request.form["sepal_width"]),
                float(request.form["petal_length"]),
                float(request.form["petal_width"]),
            ]
            # Make prediction
            prediction = predict_flower(features)
            flower_types = ["Setosa", "Versicolor", "Virginica"]
            result = flower_types[prediction]
            return render_template("index.html", result=result)

        except Exception as e:
            return render_template("index.html", result="Error: " + str(e))

    return render_template("index.html", result=None)

if __name__ == "__main__":
    app.run(debug=True)
