# 🛡️ PhishGuard - AI-Powered Phishing URL Detection System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-orange.svg)](https://www.tensorflow.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-red.svg)](https://xgboost.readthedocs.io/)
[![Flask](https://img.shields.io/badge/Flask-2.3%2B-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An advanced machine learning system for detecting phishing URLs with **97%+ accuracy** using dual AI models. Features a modern web interface, RESTful API, and comprehensive analysis tools.

![PhishGuard Demo](docs/demo.png)

## 🌟 Key Features

- **Dual Model Architecture**: Choose between CNN (97.09% accuracy) or XGBoost (96.36% accuracy)
- **Real-time Detection**: Instant URL analysis with confidence scores
- **Batch Processing**: Scan multiple URLs simultaneously
- **Feature Analysis**: Understand what makes a URL suspicious
- **Modern Web UI**: Responsive, animated interface with dark theme
- **RESTful API**: Easy integration with existing systems
- **Comprehensive Visualizations**: Confusion matrices, ROC curves, feature importance
- **Production Ready**: Optimized for deployment with error handling

## 📊 Model Performance

### 1D-CNN Model
| Metric | Score |
|--------|-------|
| Accuracy | 97.09% |
| Precision | 96.89% |
| Recall | 97.30% |
| F1-Score | 97.09% |
| ROC-AUC | 0.9965 |

### XGBoost Model
| Metric | Score |
|--------|-------|
| Accuracy | 96.36% |
| Precision | 97.89% |
| Recall | 94.77% |
| F1-Score | 96.30% |
| ROC-AUC | 0.9942 |

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
pip or conda package manager
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/phishguard.git
cd phishguard
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r webapp/requirements.txt
```

4. **Download pre-trained models**

Download the pre-trained models from [Releases](https://github.com/yourusername/phishguard/releases) and place them in:
- `CNN medesly/phishing_detector_model.keras`
- `Xgboost medelsy/phishing_detector_xgboost.json`

5. **Run the web application**
```bash
cd webapp
python app.py
```

6. **Open your browser**
```
http://localhost:5000
```

## 📁 Project Structure

```
phishguard/
├── CNN medesly/
│   ├── Phishing_URL_Detection_1D_CNN.py    # CNN training script
│   ├── phishing_detector_model.keras       # Trained CNN model
│   └── results/                            # Training visualizations
│       ├── confusion_matrix.png
│       ├── roc_curve.png
│       ├── training_history.png
│       └── results_summary.json
│
├── Xgboost medelsy/
│   ├── Phishing_URL_Detection_XGBoost.py   # XGBoost training script
│   ├── phishing_detector_xgboost.json      # Trained XGBoost model
│   └── results/                            # Training visualizations
│       ├── feature_importance.png
│       ├── shap_summary.png
│       └── results_summary.json
│
├── webapp/
│   ├── app.py                              # Flask backend
│   ├── requirements.txt                    # Python dependencies
│   ├── run.bat                             # Windows startup script
│   ├── templates/
│   │   └── index.html                      # Web interface
│   └── static/
│       ├── css/style.css
│       └── js/main.js
│
└── README.md
```

## 🔧 Usage

### Web Interface

1. **Single URL Scan**
   - Enter a URL in the input field
   - Select model (CNN or XGBoost)
   - Click "Scan URL"
   - View results with confidence score

2. **Batch Scan**
   - Click "Batch Scan" tab
   - Enter multiple URLs (one per line)
   - Select model
   - Click "Scan All URLs"
   - Download results as JSON

### API Usage

#### Check Model Status
```bash
curl http://localhost:5000/api/status
```

Response:
```json
{
  "xgboost_available": true,
  "model_accuracy": "96.36%"
}
```

#### Single URL Prediction
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"url": "http://suspicious-site.com", "model": "xgboost"}'
```

Response:
```json
{
  "url": "http://suspicious-site.com",
  "is_phishing": true,
  "probability": 0.89,
  "confidence": 0.89,
  "label": "PHISHING",
  "model": "XGBoost",
  "top_features": [
    {"name": "path_length", "importance": 0.416},
    {"name": "has_www", "importance": 0.150}
  ]
}
```

#### Batch Prediction
```bash
curl -X POST http://localhost:5000/api/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"urls": ["url1.com", "url2.com"], "model": "cnn"}'
```

### Python Integration

```python
import requests

def check_url(url):
    response = requests.post(
        'http://localhost:5000/api/predict',
        json={'url': url, 'model': 'xgboost'}
    )
    return response.json()

result = check_url('http://example.com')
print(f"Is Phishing: {result['is_phishing']}")
print(f"Confidence: {result['confidence']:.2%}")
```

## 🧠 Model Architecture

### 1D-CNN Model

```
Input (200 characters)
    ↓
Embedding Layer (64 dimensions)
    ↓
Parallel Conv1D Branches (kernel sizes: 3, 5, 7)
    ↓
Batch Normalization
    ↓
Global Max Pooling
    ↓
Concatenate
    ↓
Dense Layers (128, 64) + Dropout (0.5)
    ↓
Output (Sigmoid)
```

**Key Features:**
- Character-level analysis
- Multi-scale pattern detection
- Robust to obfuscation
- No manual feature engineering

### XGBoost Model

**47 Handcrafted Features:**
- Length-based: URL, domain, path, query lengths
- Count-based: Dots, hyphens, slashes, special characters
- Binary: IP address, HTTPS, suspicious TLD
- Entropy: URL and domain randomness
- Keywords: Login, security, brand names
- Statistical: Digit ratio, vowel ratio

**Top 5 Important Features:**
1. Path length (41.6%)
2. Has www (15.0%)
3. Number of slashes (9.0%)
4. Subdomain length (3.2%)
5. TLD length (3.1%)

## 📈 Training Your Own Models

### CNN Model

```bash
cd "CNN medesly"
python Phishing_URL_Detection_1D_CNN.py
```

**Configuration:**
- Training samples: 500,000
- Test samples: 100,000
- Epochs: 20
- Batch size: 256
- Training time: ~7 minutes (GPU)

### XGBoost Model

```bash
cd "Xgboost medelsy"
python Phishing_URL_Detection_XGBoost.py
```

**Configuration:**
- Training samples: 500,000
- Test samples: 100,000
- Estimators: 500
- Max depth: 8
- Training time: ~0.11 minutes (GPU)

## 📊 Dataset

**Mendeley Phishing Website Detection Dataset**
- Training: 4M URLs (2M benign + 2M phishing)
- Testing: 2.5M URLs (1M benign + 1.5M phishing)
- Source: [Mendeley Data](https://data.mendeley.com/)

## 🔍 Feature Engineering

The XGBoost model uses 47 features across multiple categories:

### Length Features
- URL length, domain length, path length, query length

### Count Features
- Number of dots, hyphens, slashes, digits, special characters

### Binary Features
- Has IP address, HTTPS, www, port, @ symbol

### Domain Features
- Subdomain count, TLD type, domain entropy

### Keyword Features
- Suspicious keywords (login, verify, account, etc.)
- Brand names (paypal, amazon, google, etc.)

### Statistical Features
- Shannon entropy, vowel/consonant ratios

## 🛠️ Technologies Used

- **Deep Learning**: TensorFlow, Keras
- **Machine Learning**: XGBoost, scikit-learn
- **Web Framework**: Flask, Flask-CORS
- **Data Processing**: NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn
- **Explainability**: SHAP
- **URL Parsing**: tldextract, urllib
- **Frontend**: HTML5, CSS3, JavaScript

## 📦 Dependencies

```txt
tensorflow>=2.13.0
xgboost>=2.0.0
flask>=2.3.0
flask-cors>=4.0.0
scikit-learn>=1.3.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
shap>=0.42.0
tldextract>=3.4.0
```

## 🚀 Deployment

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY webapp/ /app/
COPY "CNN medesly/phishing_detector_model.keras" /app/models/
COPY "Xgboost medelsy/phishing_detector_xgboost.json" /app/models/

RUN pip install -r requirements.txt

EXPOSE 5000
CMD ["python", "app.py"]
```

Build and run:
```bash
docker build -t phishguard .
docker run -p 5000:5000 phishguard
```

### Cloud Deployment

The application can be deployed to:
- **Heroku**: Use `Procfile` with gunicorn
- **AWS**: EC2, Lambda, or Elastic Beanstalk
- **Google Cloud**: App Engine or Cloud Run
- **Azure**: App Service

## 🧪 Testing

```bash
# Run unit tests
python -m pytest tests/

# Test API endpoints
python tests/test_api.py

# Load testing
python tests/load_test.py
```

## 📝 API Documentation

Full API documentation available at `/api/docs` when running the server.

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/status` | Check model availability |
| POST | `/api/predict` | Single URL prediction |
| POST | `/api/predict/batch` | Batch URL prediction |

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **FYP Project Team** - *Initial work* - January 2026

## 🙏 Acknowledgments

- Mendeley for providing the phishing dataset
- TensorFlow and XGBoost communities
- Research papers on phishing detection
- Open-source contributors

## 📧 Contact

For questions or support, please open an issue or contact:
- Email: your.email@example.com
- Project Link: [https://github.com/yourusername/phishguard](https://github.com/yourusername/phishguard)

## 🔮 Future Enhancements

- [ ] Real-time URL monitoring
- [ ] Browser extension
- [ ] Mobile app
- [ ] Multi-language support
- [ ] Integration with threat intelligence feeds
- [ ] Automated model retraining pipeline
- [ ] Advanced SHAP visualizations
- [ ] User feedback loop for model improvement

## 📚 Research & References

1. Character-level CNNs for text classification
2. XGBoost: A Scalable Tree Boosting System
3. Phishing website detection using URL features
4. Deep learning approaches to cybersecurity

---

**⭐ If you find this project useful, please consider giving it a star!**

Made with ❤️ for cybersecurity
