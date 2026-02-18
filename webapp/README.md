# 🛡️ PhishGuard - Phishing URL Detection Web Application

A modern, professional web application for detecting phishing URLs using AI/ML models.

## ✨ Features

- **Dual Model Support**: Choose between XGBoost (fast, interpretable) or 1D-CNN (accurate, deep learning)
- **Single URL Scan**: Analyze individual URLs with detailed results
- **Batch Scanning**: Scan multiple URLs at once
- **Real-time Results**: Instant feedback with confidence scores
- **Feature Analysis**: See what factors contributed to the prediction (XGBoost)
- **Modern UI**: Responsive, animated interface with dark theme

## 🚀 Quick Start

### Option 1: Using the Batch File (Windows)
```
Double-click run.bat
```

### Option 2: Manual Start
```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

Then open your browser and go to: **http://localhost:5000**

## 📁 Project Structure

```
webapp/
├── app.py                 # Flask backend with API endpoints
├── requirements.txt       # Python dependencies
├── run.bat               # Windows startup script
├── README.md             # This file
├── templates/
│   └── index.html        # Main HTML template
└── static/
    ├── css/
    │   └── style.css     # Styles
    └── js/
        └── main.js       # Frontend JavaScript
```

## 🔌 API Endpoints

### Check Model Status
```
GET /api/status
Response: { "cnn_available": true, "xgboost_available": true }
```

### Single URL Prediction
```
POST /api/predict
Body: { "url": "example.com", "model": "xgboost" }
Response: {
    "url": "example.com",
    "is_phishing": false,
    "probability": 0.12,
    "confidence": 0.88,
    "label": "SAFE",
    "model": "XGBoost"
}
```

### Batch Prediction
```
POST /api/predict/batch
Body: { "urls": ["url1.com", "url2.com"], "model": "cnn" }
Response: { "results": [...], "total": 2 }
```

## 🤖 Models

### XGBoost
- **Accuracy**: 96.36%
- **Precision**: 97.89%
- **Features**: 47 handcrafted URL features
- **Speed**: ~0.11s training, instant inference
- **Best for**: Fast, explainable predictions

### 1D-CNN
- **Accuracy**: 97.09%
- **Recall**: 97.30%
- **ROC-AUC**: 0.9965
- **Speed**: Slightly slower inference
- **Best for**: Maximum accuracy, character-level analysis

## 📋 Requirements

- Python 3.8+
- TensorFlow 2.13+
- XGBoost 2.0+
- Flask 2.3+

## 🔧 Configuration

Model paths are configured in `app.py`:
```python
CNN_MODEL_PATH = "../CNN/phishing_detector_model.keras"
XGBOOST_MODEL_PATH = "../Xgboost/phishing_detector_xgboost.json"
```

## 📝 License

This project is part of a Final Year Project (FYP) - January 2026

## 🎓 Credits

- **Dataset**: Mendeley Phishing Website Detection Dataset
- **Models**: TensorFlow (CNN), XGBoost
- **UI Framework**: Custom CSS with modern animations
