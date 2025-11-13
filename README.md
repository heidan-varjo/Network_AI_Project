# Network_AI_Project
AI-based real-time packet-level network error prediction system using Python, TensorFlow, and synthetic dataset generation.
# Real-Time Packet-Level AI System for Network Error Prediction  
### Author: Atri Pramanik (23BCE0200)  
### Course: Network Systems & AI Integration – Lab Project  

---

## 📘 Overview  

This project implements a **real-time AI-based packet-level network error prediction system** using:  
- Python  
- TensorFlow / Keras  
- Scikit-learn  
- Synthetic dataset generation  

The system predicts whether an incoming packet will be **ERRONEOUS** or **NORMAL**, based on 12 engineered network features including packet size, latency, queue depth, congestion score, protocol type, and more.

The solution integrates:  
✔ Dataset generator  
✔ Neural network training pipeline  
✔ Real-time inference simulator  
✔ Result visualizations and metrics  

---

## 🏗 Project Structure  
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

