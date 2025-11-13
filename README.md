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

```plaintext
📁 Project Structure

Network_AI_Project/
│
├── packet_data_generator.py
├── train_model.py
├── realtime_demo.py
├── requirements.txt
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
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── training_history.png
│
└── runs/
    └── 2025xxxxx/
        ├── network_error_model.h5
        ├── scaler.pkl
        ├── packet_data_full.csv
        └── training_history.png


---

# ✔ VERY IMPORTANT — READ THIS  
The VERY FIRST and VERY LAST lines MUST be:

```plaintext


## 🚀 Installation  

### **1. Clone Repository**


git clone https://github.com/heidan-varjo/Network_AI_Project

cd Network_AI_Project


### **2. Create Virtual Environment**


python -m venv .venv
..venv\Scripts\activate # For Windows PowerShell


### **3. Install Dependencies**


pip install -r requirements.txt


---

## 🧪 1. Generate Dataset



python packet_data_generator.py --total_samples 100000


Output files will appear in `archive/best_run/` after processing.

---

## 📘 2. Train the Model



python train_model.py --epochs 50 --batch-size 32


This will produce:  
- network_error_model.h5  
- confusion_matrix.png  
- roc_curve.png  
- training_history.png  
- scaler.pkl  
- feature_columns.json  

---

## ⚡ 3. Real-Time Prediction Demo  



python realtime_demo.py


The script loads the model + scaler and predicts packet errors with **~1.2 ms latency per packet**.

---

## 📊 Performance Summary  

- Test accuracy: **94.8%**  
- Precision (error class): **93.1%**  
- Recall (error class): **90.4%**  
- F1 Score: **91.7%**  
- ROC–AUC: **0.973**  
- Real-time inference delay: **~1.2 ms**  

---

## 📝 License  

This project is licensed under the **MIT License**.

---

## 👤 Author  

**Atri Pramanik**  
Reg No: 23BCE0200  
B.Tech CSE Core  
Vellore Institute of Technology (VIT), Vellore  

