# 🌾 Smart Farming Crop Recommendation System (Machine Learning)

[![Python Version](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask Framework](https://img.shields.io/badge/Flask-2.3%2B-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Model Accuracy](https://img.shields.io/badge/Model%20Accuracy-99.55%25-2d6a4f?style=for-the-badge&logo=target)](https://github.com/AJANYA-K-A/Smart-farming-crop-recommendation-system-Machine-learning)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)

An intelligent, data-driven precision agriculture web application that predicts the most suitable crop to cultivate based on specific soil macronutrients and environmental climate parameters.

---

## 📌 Project Overview

Traditional crop selection often relies on historical heuristics that may fail under changing climatic conditions or depleted soil nutrient profiles. The **Smart Farming Crop Recommendation System** bridges the gap between agricultural science and artificial intelligence by employing an ensemble **Random Forest Classifier** trained on 2,200 agricultural field records.

By analyzing 7 critical agronomic parameters (Nitrogen, Phosphorus, Potassium, Temperature, Humidity, Soil pH, and Rainfall), the system provides:
1. **Accurate Crop Recommendations** with confidence scores.
2. **Comprehensive Agronomic Profiles** (ideal season, water requirement, soil compatibility, and farming tips).
3. **Exploratory Visual Analytics** of soil chemistry and climate distributions.
4. **Interactive FarmBot AI Assistant** to answer agricultural, nutrient, and irrigation queries.
5. **RESTful JSON API** for integration with IoT farm sensors and mobile applications.

---

## ✨ Key Features

- **🔐 Dedicated Authentication Portal**: Professional login screen with 1-Click Demo Access buttons ("🌱 Enter as Farmer", "🔬 Enter as Agronomist", "🚀 Quick Guest Access") and secure session management.
- **📊 Executive Precision Farming Dashboard**: Tabbed interface featuring real-time KPI metrics, active crop recommendation engine, dataset analytics, and agronomy guides.
- **🎯 High Accuracy ML Engine**: Trained Random Forest model achieving **99.55% test accuracy** across 22 diverse agricultural crops.
- **⚡ 1-Click Test Presets**: Pre-configured real-world profiles for Rice (Wetlands), Maize (Plains), Coffee (Hills), Watermelon (Sandy), Apple (Temperate), and Cotton (Black Soil).
- **🧪 Interactive Soil & Climate Inputs**: Dual-synchronized range sliders and precision numeric fields with accessible keyboard support.
- **🌱 Rich Agronomic Insights**: Displays crop category, growing season, water demand, soil compatibility, and agronomist tips for each recommendation.
- **🤖 FarmBot Agricultural Chatbot**: In-app conversational assistant linked directly to the backend to guide users on NPK nutrients, soil pH adjustments, and smart irrigation.
- **📊 Dataset Visual Analytics**: Built-in visual distributions (histograms, scatter correlations, and heatmaps) with one-click lightbox zoom.
- **🌐 RESTful API**: Production endpoints (`/api/predict`, `/chat`, `/api/health`, `/api/crops`) ready for headless deployment or mobile clients.
- **🚀 Production Ready**: Fully configured for local execution and cloud deployment (Render, Railway, Heroku, or Docker).

---

## 🧠 Supported Crops (22 Classes)

The model recommends optimal conditions for 22 staple, cash, pulse, fruit, and plantation crops:

| Category | Crops |
| :--- | :--- |
| **Cereals & Grains** | Rice (`rice`), Maize (`maize`) |
| **Pulses & Legumes** | Chickpea (`chickpea`), Kidney Beans (`kidneybeans`), Pigeon Peas (`pigeonpeas`), Moth Beans (`mothbeans`), Mung Bean (`mungbean`), Black Gram (`blackgram`), Lentil (`lentil`) |
| **Fruits** | Pomegranate (`pomegranate`), Banana (`banana`), Mango (`mango`), Grapes (`grapes`), Watermelon (`watermelon`), Muskmelon (`muskmelon`), Apple (`apple`), Orange (`orange`), Papaya (`papaya`) |
| **Plantation & Cash** | Coconut (`coconut`), Cotton (`cotton`), Jute (`jute`), Coffee (`coffee`) |

---

## 📊 Dataset & Features

The dataset comprises **2,200 instances** with 7 numerical features and 1 categorical target:

| Feature | Name | Unit | Typical Range | Description |
| :--- | :--- | :--- | :--- | :--- |
| **N** | Nitrogen | kg/ha | `0 – 140` | Ratio of Nitrogen content in soil |
| **P** | Phosphorus | kg/ha | `5 – 145` | Ratio of Phosphorus content in soil |
| **K** | Potassium | kg/ha | `5 – 205` | Ratio of Potassium content in soil |
| **temperature** | Temperature | °C | `8.8 – 43.7` | Ambient ambient temperature |
| **humidity** | Relative Humidity | % | `14.2 – 99.9` | Relative atmospheric humidity |
| **ph** | Soil pH | pH scale | `3.5 – 9.9` | Acidity / alkalinity value of the soil |
| **rainfall** | Rainfall | mm | `20.2 – 298.6` | Average annual precipitation |

---

## 🔬 Model Performance & Evaluation

The model was evaluated using an 80/20 stratified split:

```
[+] Model Evaluation Accuracy: 99.55%

Detailed Classification Report:
              precision    recall  f1-score   support
       apple       1.00      1.00      1.00        20
      banana       1.00      1.00      1.00        20
   blackgram       1.00      0.95      0.97        20
    chickpea       1.00      1.00      1.00        20
     coconut       1.00      1.00      1.00        20
      coffee       1.00      1.00      1.00        20
      cotton       1.00      1.00      1.00        20
      grapes       1.00      1.00      1.00        20
        jute       0.95      1.00      0.98        20
 kidneybeans       1.00      1.00      1.00        20
      lentil       1.00      1.00      1.00        20
       maize       0.95      1.00      0.98        20
       mango       1.00      1.00      1.00        20
   mothbeans       1.00      1.00      1.00        20
    mungbean       1.00      1.00      1.00        20
   muskmelon       1.00      1.00      1.00        20
      orange       1.00      1.00      1.00        20
      papaya       1.00      1.00      1.00        20
  pigeonpeas       1.00      1.00      1.00        20
 pomegranate       1.00      1.00      1.00        20
        rice       1.00      0.95      0.97        20
  watermelon       1.00      1.00      1.00        20

    accuracy                           1.00       440
   macro avg       1.00      1.00      1.00       440
weighted avg       1.00      1.00      1.00       440
```

---

## 📁 Repository Structure

```
Smart-farming-crop-recommendation-system-Machine-learning/
│
├── Crop_recommendation.csv    # Benchmark dataset (2200 samples)
├── crop_model.pkl             # Serialized Random Forest model artifact
├── label_encoder.pkl          # Serialized label encoder artifact
├── train_model.py             # Reproducible training & evaluation script
├── app.py                     # Flask backend, APIs, and FarmBot service
├── test_app.py                # Automated integration & unit test suite
├── requirements.txt           # Python dependencies with pinned versions
├── .gitignore                 # Standard Python & OS ignore rules
├── README.md                  # Comprehensive project documentation
├── templates/
│   ├── login.html             # Dedicated authentication portal with 1-click access
│   ├── register.html          # Dedicated registration & sign-up portal
│   ├── logout.html            # Dedicated secure sign-out confirmation page
│   ├── dashboard.html         # Executive precision farming dashboard with tabs & KPIs
│   └── index.html             # Standalone fallback dashboard interface
│
├── static/
│   ├── chart1.png             # Nitrogen distribution plot
│   ├── chart2.png             # Phosphorus distribution plot
│   ├── chart3.png             # Potassium distribution plot
│   ├── chart4.png             # Temperature distribution plot
│   ├── ...                    # Additional exploratory charts
│   └── chart10.png            # Feature correlation heatmap
│
└── output/
    ├── random_forest_tree.png               # Visualized decision tree
    └── random_forest_tree_with_levels.png   # Decision tree with labeled levels
```

---

## 🚀 Quickstart & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/AJANYA-K-A/Smart-farming-crop-recommendation-system-Machine-learning.git
cd Smart-farming-crop-recommendation-system-Machine-learning
```

### 2. Set Up a Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. (Optional) Retrain the Model
To re-evaluate the model from source data:
```bash
python train_model.py
```

### 5. Run the Automated Tests
```bash
python test_app.py
```

### 6. Start the Web Application
```bash
python app.py
```
Open your browser and navigate to: **`http://127.0.0.1:5000`**

---

## 📡 REST API Reference

The application includes headless REST endpoints for programmatic access and IoT integrations:

### 1. Predict Crop
- **Endpoint**: `POST /api/predict`
- **Headers**: `Content-Type: application/json`
- **Request Body**:
```json
{
  "nitrogen": 90,
  "phosphorus": 42,
  "potassium": 43,
  "temperature": 20.88,
  "humidity": 82.0,
  "ph": 6.5,
  "rainfall": 202.9
}
```
- **Response**:
```json
{
  "status": "success",
  "recommended_crop": "rice",
  "confidence_percent": 98.0,
  "crop_details": {
    "category": "Cereal Grain",
    "emoji": "🌾",
    "season": "Kharif (Monsoon)",
    "water": "High (Standing water, 150-300 mm rainfall)",
    "soil": "Clayey or loamy soil with pH 5.5–7.0",
    "temp": "20°C – 35°C",
    "tip": "Maintain consistent puddle water during tillering and panicle development."
  }
}
```

### 2. Chat with FarmBot
- **Endpoint**: `POST /chat`
- **Request Body**:
```json
{
  "message": "What is the best soil pH for farming?"
}
```
- **Response**:
```json
{
  "reply": "Most agricultural crops prefer a soil pH between 6.0 and 7.2. For acidic soils (<6.0), add agricultural lime (calcium carbonate). For alkaline soils (>7.5), incorporate gypsum or elemental sulfur."
}
```

### 3. Health Check
- **Endpoint**: `GET /api/health`
- **Response**:
```json
{
  "status": "healthy",
  "service": "Smart Farming Crop Recommendation API",
  "model_loaded": true,
  "supported_crops": 22
}
```

---

---

## 📄 License

This project is licensed under the **MIT License** — feel free to use, modify, and distribute it for personal, research, or commercial purposes.

---

## 👨‍💻 Author & Acknowledgments

- **Author**: [AJANYA K A](https://github.com/AJANYA-K-A)
- **Repository**: [Smart-farming-crop-recommendation-system-Machine-learning](https://github.com/AJANYA-K-A/Smart-farming-crop-recommendation-system-Machine-learning)
- **Acknowledgments**: Built with Scikit-Learn, Flask, Pandas, and Matplotlib. Dedicated to empowering farmers with data-driven precision tools.
