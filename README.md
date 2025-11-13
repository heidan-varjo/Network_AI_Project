Real-Time Packet-Level AI System for Network Error Prediction

Author: Atri Pramanik (23BCE0200)
Course: Network Systems and AI Integration – Lab Project

Overview

This project implements a real-time AI-based packet-level network error prediction system.
It predicts whether a network packet is normal or erroneous using:

Python

TensorFlow / Keras

Scikit-learn

Synthetic dataset generation

Real-time inference simulation

The project includes dataset generation, neural network training, model evaluation, visualization, and real-time inference pipeline.

Project Structure
Network_AI_Project/
│
├── archive/                         
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
├── images/                          
│   ├── training_history.png
│   ├── confusion_matrix.png
│   └── roc_curve.png
│
├── runs/                            
│   └── 2025.../                     
│
├── packet_data_generator.py         
├── train_model.py                   
├── realtime_demo.py                 
├── requirements.txt                 
└── README.md                        

Features
Synthetic Packet Dataset Generator

Generates 100,000 realistic packet samples.

Models queueing delays, congestion levels, burst errors.

Uses 12 engineered numerical features.

Automatically creates train/validation/test splits.

Neural Network Model

Architecture: Dense layers 128 → 64 → 32 → 1.

Dropout regularization to reduce overfitting.

Binary classification using sigmoid activation.

Performance:

Accuracy: 94.8%

Error-class recall: 90.4%

F1-score: 91.7%

ROC-AUC: 0.973

Real-Time Inference Simulation

Loads trained model and scaler.

Predicts packet error probability.

Average inference time: about 1.2 ms on CPU.

Demonstrates CPU-optimized real-time prediction.

Installation
1. Clone the repository
git clone https://github.com/yourusername/Network_AI_Project.git
cd Network_AI_Project

2. Create a virtual environment (optional)

Windows:

python -m venv venv
venv\Scripts\activate


Linux/Mac:

python3 -m venv venv
source venv/bin/activate

3. Install required libraries
pip install -r requirements.txt

How to Run
1. Generate the dataset
python packet_data_generator.py


This generates:

packet_data_train.csv

packet_data_val.csv

packet_data_test.csv

packet_data_full.csv

feature_columns.json

Files are stored inside archive/.

2. Train the model
python train_model.py


This produces:

network_error_model.h5

scaler.pkl

model_metrics.json

training_history.png

confusion_matrix.png

roc_curve.png

All stored in archive/.

3. Run the real-time prediction demo
python realtime_demo.py


Shows prediction probability and packet latency behavior.

Example Prediction Code
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

Requirements

Dependencies inside requirements.txt:

tensorflow==2.10.0

numpy

pandas

scikit-learn

matplotlib

seaborn

Screenshots / Visualizations

All key plots are stored in:

images/

archive/

License

This project is licensed under the MIT License.
You may modify and use the code for academic or personal work.

Author

Atri Pramanik
Dept. of Computer Science
VIT Vellore
Email: atri.pramanik2023@vitstudent.ac.in
