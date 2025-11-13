Network_AI_Project

A real-time packet-level AI system for predicting network errors using Python, TensorFlow, scikit-learn, and a synthetic dataset generation pipeline.

This project demonstrates how machine learning can enhance traditional network reliability mechanisms by predicting packet-level errors before they occur. It includes dataset generation, feature engineering, neural network training, performance evaluation, and real-time packet inference simulation.

📌 Project Structure
Network_AI_Project/
│
├── archive/                 # Best run artifacts stored safely
│   ├── network_error_model.h5
│   ├── scaler.pkl
│   ├── feature_columns.json
│   ├── model_metrics.json
│   ├── packet_data_train.csv
│   ├── packet_data_test.csv
│   ├── packet_data_val.csv
│   ├── packet_data_full.csv
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── training_history.png
│
├── images/                  # Images for documentation/report
│   ├── training_history.png
│   ├── confusion_matrix.png
│   └── roc_curve.png
│
├── runs/                    # Example best run folder (only one kept)
│   └── 2025.../             # Optional: contains same artifacts as archive
│
├── packet_data_generator.py # Synthetic dataset creation (100k packets)
├── train_model.py           # Neural network model training
├── realtime_demo.py         # Real-time prediction simulation
├── requirements.txt         # Libraries needed to run the project
└── README.md                # Documentation (this file)

📌 Features
✔ Synthetic Packet Dataset Generator

Generates 100,000 realistic packet samples

Models queueing delays, congestion, burst errors

Includes 12 engineered features

Produces train/validation/test splits automatically

✔ Neural Network Model

Lightweight 128–64–32 architecture

Dropout regularization

Binary classification (normal/error)

Trained using TensorFlow/Keras

Achieves:

Accuracy: 94.8%

Recall (Error Class): 90.4%

F1 Score: 91.7%

ROC-AUC: 0.973

✔ Real-Time Inference Simulation

Loads trained model and scaler

Predicts packet error probability

Average inference latency ~ 1.2 ms (CPU)

Demonstrates CPU-optimized model behavior

📌 Installation
1. Clone the repository
git clone https://github.com/yourusername/Network_AI_Project.git
cd Network_AI_Project

2. Create a virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate     # Windows
source venv/bin/activate # Linux/Mac

3. Install dependencies
pip install -r requirements.txt

📌 How to Run the Project
1. Generate Dataset
python packet_data_generator.py


Outputs:

packet_data_train.csv

packet_data_val.csv

packet_data_test.csv

packet_data_full.csv

feature_columns.json

Stored in archive/ after processing.

2. Train the Neural Network
python train_model.py


Outputs:

network_error_model.h5

scaler.pkl

model_metrics.json

training_history.png

confusion_matrix.png

roc_curve.png

Saved inside archive/.

3. Run Real-Time Prediction Demo
python realtime_demo.py


Shows:

latency per packet

prediction score

threshold decision

📌 Example Prediction Code
import tensorflow as tf
import numpy as np
import json, pickle

model = tf.keras.models.load_model("archive/network_error_model.h5")
scaler = pickle.load(open("archive/scaler.pkl", "rb"))
cols = json.load(open("archive/feature_columns.json"))

def predict(features):
    x = np.array([[features[c] for c in cols]])
    x_scaled = scaler.transform(x)
    prob = model.predict(x_scaled)[0, 0]
    return prob

📌 Requirements

Main dependencies (full list in requirements.txt):

tensorflow==2.10.0
numpy
pandas
scikit-learn
matplotlib
seaborn

📌 Screenshots / Visualizations

All key plots are included inside:

images/

📌 License

This project is licensed under the MIT License — free to modify and use for academic or personal work.

📌 Author

Atri Pramanik (23BCE0200)
Department of Computer Science
VIT Vellore
Email: atri.pramanik2023@vitstudent.ac.in
