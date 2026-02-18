"""
================================================================================
PHISHING URL DETECTION WEB APPLICATION
================================================================================
A Flask-based web application that provides phishing URL detection using
XGBoost model with 96.36% accuracy.

Author: FYP Project
Date: January 2026
================================================================================
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import warnings
import re
import math
from urllib.parse import urlparse
from collections import Counter

warnings.filterwarnings('ignore')

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# ================================================================================
# MODEL PATHS
# ================================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
XGBOOST_MODEL_PATH = os.path.join(PARENT_DIR, "Xgboost", "phishing_detector_xgboost.json")

# ================================================================================
# GLOBAL VARIABLES
# ================================================================================
xgboost_model = None
feature_extractor = None

# ================================================================================
# TLD EXTRACT SETUP
# ================================================================================
try:
    import tldextract
    tldextract.extract("test.com")  # Pre-cache
except ImportError:
    print("Warning: tldextract not installed. Installing...")
    os.system(f"{sys.executable} -m pip install tldextract")
    import tldextract


# ================================================================================
# FEATURE EXTRACTOR FOR XGBOOST (EXACT MATCH WITH TRAINING)
# ================================================================================
class URLFeatureExtractor:
    """
    Extract features from URLs for phishing detection.
    Features are based on research literature and domain knowledge.
    MUST MATCH EXACTLY with training script features.
    """
    
    # Suspicious keywords commonly found in phishing URLs
    SUSPICIOUS_KEYWORDS = [
        'login', 'signin', 'sign-in', 'log-in', 'account', 'update', 'secure',
        'banking', 'confirm', 'verify', 'verification', 'password', 'credential',
        'suspend', 'restrict', 'unlock', 'authenticate', 'wallet', 'paypal',
        'ebay', 'amazon', 'apple', 'microsoft', 'google', 'facebook', 'netflix',
        'bank', 'credit', 'debit', 'ssn', 'social', 'tax', 'refund', 'prize',
        'winner', 'lucky', 'free', 'gift', 'bonus', 'offer', 'click', 'urgent',
        'immediately', 'expire', 'limited', 'act-now', 'alert', 'warning'
    ]
    
    # Suspicious TLDs often used in phishing
    SUSPICIOUS_TLDS = [
        'tk', 'ml', 'ga', 'cf', 'gq', 'xyz', 'top', 'pw', 'cc', 'su',
        'buzz', 'work', 'party', 'gdn', 'kim', 'country', 'stream',
        'download', 'racing', 'win', 'bid', 'loan', 'men', 'click'
    ]
    
    def __init__(self):
        self.feature_names = []
        
    def extract_features(self, url):
        """Extract all features from a single URL - MUST MATCH TRAINING EXACTLY."""
        features = {}
        
        # Ensure URL has a scheme for parsing
        if not url.startswith(('http://', 'https://')):
            url_with_scheme = 'http://' + url
        else:
            url_with_scheme = url
        
        try:
            parsed = urlparse(url_with_scheme)
            extracted = tldextract.extract(url)
        except:
            parsed = None
            extracted = None
        
        # === LENGTH-BASED FEATURES ===
        features['url_length'] = len(url)
        features['domain_length'] = len(extracted.domain) if extracted else 0
        features['path_length'] = len(parsed.path) if parsed else 0
        features['query_length'] = len(parsed.query) if parsed else 0
        features['fragment_length'] = len(parsed.fragment) if parsed else 0
        
        # === COUNT-BASED FEATURES ===
        features['num_dots'] = url.count('.')
        features['num_hyphens'] = url.count('-')
        features['num_underscores'] = url.count('_')
        features['num_slashes'] = url.count('/')
        features['num_question_marks'] = url.count('?')
        features['num_equals'] = url.count('=')
        features['num_at'] = url.count('@')
        features['num_ampersand'] = url.count('&')
        features['num_percent'] = url.count('%')
        features['num_digits'] = sum(c.isdigit() for c in url)
        features['num_letters'] = sum(c.isalpha() for c in url)
        features['num_special_chars'] = sum(not c.isalnum() for c in url)
        features['num_subdomains'] = len(extracted.subdomain.split('.')) if extracted and extracted.subdomain else 0
        
        # === RATIO FEATURES ===
        url_len = len(url) if len(url) > 0 else 1
        features['digit_ratio'] = features['num_digits'] / url_len
        features['letter_ratio'] = features['num_letters'] / url_len
        features['special_char_ratio'] = features['num_special_chars'] / url_len
        
        # === BINARY FEATURES ===
        features['has_ip_address'] = 1 if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url) else 0
        features['has_https'] = 1 if url.startswith('https') else 0
        features['has_http'] = 1 if url.startswith('http://') else 0
        features['has_www'] = 1 if 'www.' in url.lower() else 0
        features['has_port'] = 1 if re.search(r':\d+', url) else 0
        features['has_at_symbol'] = 1 if '@' in url else 0
        features['has_double_slash'] = 1 if '//' in url[8:] else 0  # After protocol
        features['has_hex_chars'] = 1 if re.search(r'%[0-9a-fA-F]{2}', url) else 0
        
        # === DOMAIN FEATURES ===
        if extracted:
            features['domain_has_digits'] = 1 if any(c.isdigit() for c in extracted.domain) else 0
            features['domain_has_hyphens'] = 1 if '-' in extracted.domain else 0
            features['subdomain_length'] = len(extracted.subdomain) if extracted.subdomain else 0
            features['tld_length'] = len(extracted.suffix) if extracted.suffix else 0
            features['is_suspicious_tld'] = 1 if extracted.suffix.lower() in self.SUSPICIOUS_TLDS else 0
        else:
            features['domain_has_digits'] = 0
            features['domain_has_hyphens'] = 0
            features['subdomain_length'] = 0
            features['tld_length'] = 0
            features['is_suspicious_tld'] = 0
        
        # === ENTROPY FEATURES ===
        features['url_entropy'] = self._calculate_entropy(url)
        features['domain_entropy'] = self._calculate_entropy(extracted.domain) if extracted else 0
        
        # === KEYWORD FEATURES ===
        url_lower = url.lower()
        features['num_suspicious_keywords'] = sum(1 for kw in self.SUSPICIOUS_KEYWORDS if kw in url_lower)
        features['has_login_keyword'] = 1 if any(kw in url_lower for kw in ['login', 'signin', 'log-in', 'sign-in']) else 0
        features['has_security_keyword'] = 1 if any(kw in url_lower for kw in ['secure', 'account', 'update', 'verify']) else 0
        features['has_brand_keyword'] = 1 if any(kw in url_lower for kw in ['paypal', 'ebay', 'amazon', 'apple', 'microsoft', 'google', 'facebook', 'netflix', 'bank']) else 0
        
        # === PATH FEATURES ===
        if parsed and parsed.path:
            path_parts = [p for p in parsed.path.split('/') if p]
            features['path_depth'] = len(path_parts)
            features['max_path_segment_length'] = max(len(p) for p in path_parts) if path_parts else 0
            features['avg_path_segment_length'] = np.mean([len(p) for p in path_parts]) if path_parts else 0
        else:
            features['path_depth'] = 0
            features['max_path_segment_length'] = 0
            features['avg_path_segment_length'] = 0
        
        # === QUERY FEATURES ===
        if parsed and parsed.query:
            query_params = parsed.query.split('&')
            features['num_query_params'] = len(query_params)
            features['avg_query_param_length'] = np.mean([len(p) for p in query_params])
        else:
            features['num_query_params'] = 0
            features['avg_query_param_length'] = 0
        
        # === STATISTICAL FEATURES ===
        features['vowel_ratio'] = sum(1 for c in url.lower() if c in 'aeiou') / url_len
        features['consonant_ratio'] = sum(1 for c in url.lower() if c.isalpha() and c not in 'aeiou') / url_len
        
        # Store feature names on first call
        if not self.feature_names:
            self.feature_names = list(features.keys())
        
        return features
    
    def _calculate_entropy(self, text):
        """Calculate Shannon entropy of a string."""
        if not text:
            return 0
        freq = Counter(text)
        length = len(text)
        entropy = 0
        for count in freq.values():
            prob = count / length
            if prob > 0:
                entropy -= prob * math.log2(prob)
        return entropy


# ================================================================================
# MODEL LOADING
# ================================================================================
def load_xgboost_model():
    """Load the XGBoost model."""
    global xgboost_model
    try:
        import xgboost as xgb
        
        if os.path.exists(XGBOOST_MODEL_PATH):
            xgboost_model = xgb.Booster()
            xgboost_model.load_model(XGBOOST_MODEL_PATH)
            print(f"✓ XGBoost model loaded from: {XGBOOST_MODEL_PATH}")
            return True
        else:
            print(f"✗ XGBoost model not found at: {XGBOOST_MODEL_PATH}")
            return False
    except Exception as e:
        print(f"✗ Error loading XGBoost model: {e}")
        return False


def initialize_models():
    """Initialize model and feature extractor."""
    global feature_extractor
    
    print("\n" + "="*60)
    print("INITIALIZING PHISHING DETECTION MODEL")
    print("="*60)
    
    feature_extractor = URLFeatureExtractor()
    print("✓ Feature extractor initialized")
    
    xgb_loaded = load_xgboost_model()
    
    print("="*60)
    print(f"XGBoost Model: {'Ready ✓' if xgb_loaded else 'Not Available ✗'}")
    print(f"Model Accuracy: 96.36%")
    print("="*60 + "\n")
    
    return xgb_loaded


# ================================================================================
# PREDICTION FUNCTION
# ================================================================================
def predict_url(url):
    """Make prediction using XGBoost model."""
    global xgboost_model, feature_extractor
    
    if xgboost_model is None:
        return None, "XGBoost model not loaded"
    
    try:
        import xgboost as xgb
        
        # Extract features
        features = feature_extractor.extract_features(url)
        X = pd.DataFrame([features])
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Use DMatrix for prediction with Booster
        dmatrix = xgb.DMatrix(X)
        probability = float(xgboost_model.predict(dmatrix)[0])
        
        is_phishing = probability >= 0.5
        confidence = probability if is_phishing else 1 - probability
        
        # Get feature importance from booster
        try:
            importance_dict = xgboost_model.get_score(importance_type='gain')
            top_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:5]
        except:
            top_features = []
        
        return {
            'url': url,
            'is_phishing': is_phishing,
            'probability': probability,
            'confidence': confidence,
            'label': 'PHISHING' if is_phishing else 'SAFE',
            'model': 'XGBoost',
            'top_features': [{'name': f[0], 'importance': float(f[1])} for f in top_features],
            'url_features': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                           for k, v in list(features.items())[:10]}
        }, None
        
    except Exception as e:
        return None, str(e)


# ================================================================================
# FLASK ROUTES
# ================================================================================
@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')


@app.route('/api/status')
def api_status():
    """Check model status."""
    return jsonify({
        'xgboost_available': xgboost_model is not None,
        'model_accuracy': '96.36%'
    })


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for URL prediction."""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        url = data.get('url', '').strip()
        
        if not url:
            return jsonify({'error': 'URL is required'}), 400
        
        result, error = predict_url(url)
        
        if error:
            return jsonify({'error': error}), 500
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict/batch', methods=['POST'])
def api_predict_batch():
    """API endpoint for batch URL prediction."""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        urls = data.get('urls', [])
        
        if not urls:
            return jsonify({'error': 'URLs list is required'}), 400
        
        results = []
        for url in urls[:100]:  # Limit to 100 URLs
            url = url.strip()
            if url:
                result, error = predict_url(url)
                if result:
                    results.append(result)
                else:
                    results.append({'url': url, 'error': error})
        
        return jsonify({'results': results, 'total': len(results)})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ================================================================================
# MAIN
# ================================================================================
if __name__ == '__main__':
    initialize_models()
    
    print("\n🚀 Starting Phishing URL Detection Web Application...")
    print("📍 Open your browser and go to: http://localhost:5000")
    print("Press Ctrl+C to stop the server\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
