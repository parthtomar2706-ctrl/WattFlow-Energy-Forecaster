# WattFlow: SmartGrid Hourly Energy Load Forecasting

A deep learning time-series regression pipeline built with TensorFlow/Keras to predict electrical power demand using Stacked Long Short-Term Memory (LSTM) networks.

## 🧠 Architecture Overview
* **Input Layer:** 60-step sliding temporal window (`[samples, 60, 1]`).
* **Hidden Layers:** 
  * Stacked LSTM (64 units, `return_sequences=True`)
  * Dropout Regularization (`0.2`)
  * Stacked LSTM (32 units, `return_sequences=False`)
  * Dropout Regularization (`0.2`)
  * Dense Regression Layer (25 units, ReLU)
* **Output Layer:** Single continuous regression neuron predicting load in Megawatts (MW).

## 🛠️ Tech Stack
* **Language:** Python
* **Deep Learning:** TensorFlow / Keras
* **Data Processing:** NumPy, Pandas, Scikit-Learn
* **Visualization:** Matplotlib

## 🚀 Getting Started Locally

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/WattFlow-Energy-Forecaster.git](https://github.com/YOUR_USERNAME/WattFlow-Energy-Forecaster.git)
   cd WattFlow-Energy-Forecaster
