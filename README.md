### README.md

```markdown
# Iris Flower Classification Flask App 🌸

This repository contains a Flask-based web application for classifying Iris flowers into one of three species: **Setosa**, **Versicolor**, or **Virginica**. The backend is powered by a PyTorch-trained neural network model.

## Features

- **Interactive Web Interface**: Users can input flower measurements (sepal length, sepal width, petal length, petal width) and get predictions instantly.
- **PyTorch Model Integration**: The app uses a pre-trained PyTorch model saved in `.pth` format.
- **Enhanced Web Design**: Includes a responsive and clean UI with HTML and CSS.

---

## Directory Structure

```
flask_app/
│
├── static/                # Contains CSS files
│   └── styles.css         # Styling for the webpage
│
├── templates/             # Contains HTML files
│   └── index.html         # Main HTML page
│
├── iris_model.pth         # Pre-trained PyTorch model
├── app.py                 # Main Flask application file
├── requirements.txt       # Python dependencies
└── README.md              # Documentation
```

---

## How to Use

### 1. Clone the Repository
```bash
git clone https://github.com/YashashKR/iris-flower-classification-flask.git
cd iris-flower-classification-flask
```

### 2. Set Up a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Flask Application
```bash
python app.py
```

### 5. Access the Web App
Open your browser and go to [http://127.0.0.1:5000](http://127.0.0.1:5000).

---

## Model Details

The Iris classification model is a PyTorch-based feedforward neural network with the following architecture:

- Input size: 4 (sepal length, sepal width, petal length, petal width)
- Two hidden layers:
  - Hidden Layer 1: 32 neurons, ReLU activation
  - Hidden Layer 2: 16 neurons, ReLU activation, dropout (30%)
- Output size: 3 (Setosa, Versicolor, Virginica)
- Output activation: Softmax

The model was trained on the Iris dataset using:
- Loss Function: CrossEntropyLoss
- Optimizer: Adam
- Training Epochs: 100

The final trained model is saved as `iris_model.pth`.

---


## Requirements

- Python 3.7+
- Flask
- PyTorch

---

## Error Handling

The application includes basic error handling to:
1. Validate user inputs.
2. Catch server-side errors and display user-friendly messages.

---

## Future Enhancements

- Add support for uploading CSV files for batch predictions.
- Dockerize the application for easier deployment.
- Extend the UI for better user experience.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgements

- **Dataset**: [Iris Dataset](https://archive.ics.uci.edu/ml/datasets/iris) from the UCI Machine Learning Repository.
- **Frameworks**: Flask, PyTorch
``` 

