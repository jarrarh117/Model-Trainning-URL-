"""
================================================================================
PHISHING URL DETECTION USING XGBOOST
================================================================================
Author: FYP Project
Date: January 2026
Description: Complete end-to-end pipeline for detecting phishing URLs using
             XGBoost with handcrafted URL features.
             
Dataset: Mendeley Phishing Website Detection Dataset
         - 4M training URLs (2M benign + 2M phishing)
         - 2.5M test URLs (1M benign + 1.5M phishing)
         
Environment: Google Colab with GPU acceleration
================================================================================
"""

# ==============================================================================
# SECTION 1: ENVIRONMENT SETUP AND IMPORTS
# ==============================================================================

print("=" * 80)
print("SECTION 1: ENVIRONMENT SETUP AND IMPORTS")
print("=" * 80)

# Install required packages
import subprocess
import sys

def install_package(package):
    """Install a package using pip if not already installed."""
    try:
        __import__(package.split('[')[0].replace('-', '_'))
        print(f"✓ {package} already installed")
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
        print(f"✓ {package} installed successfully")

# Install necessary packages
packages_to_install = ['gdown', 'xgboost', 'scikit-learn', 'shap', 'seaborn', 'pandas', 'tldextract']
for pkg in packages_to_install:
    install_package(pkg)

# Core imports
import os
import zipfile
import warnings
import random
import time
import re
import math
from datetime import datetime
from urllib.parse import urlparse
from collections import Counter

# Data handling
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

# XGBoost
import xgboost as xgb

# Feature extraction
import tldextract
tldextract.extract("test.com")  # Pre-cache to avoid warnings during processing

# Explainability
import shap

# Suppress warnings
warnings.filterwarnings('ignore')

print("\n✓ All imports successful!")

# ==============================================================================
# SECTION 2: GPU CONFIGURATION AND SYSTEM CHECK
# ==============================================================================

print("\n" + "=" * 80)
print("SECTION 2: GPU CONFIGURATION AND SYSTEM CHECK")
print("=" * 80)

def check_gpu_for_xgboost():
    """Check if GPU is available for XGBoost."""
    print("\n--- GPU Status for XGBoost ---")
    
    try:
        # Check if CUDA is available
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ NVIDIA GPU detected")
            print(result.stdout.split('\n')[8])  # GPU info line
            # XGBoost 2.0+ uses device="cuda" instead of tree_method="gpu_hist"
            return 'hist', 'cuda'
        else:
            print("⚠ No GPU detected, using CPU")
            return 'hist', 'cpu'
    except FileNotFoundError:
        print("⚠ nvidia-smi not found, using CPU")
        return 'hist', 'cpu'

def check_system_resources():
    """Check available system resources."""
    print("\n--- System Resources ---")
    
    try:
        import psutil
        ram = psutil.virtual_memory()
        print(f"Total RAM: {ram.total / (1024**3):.2f} GB")
        print(f"Available RAM: {ram.available / (1024**3):.2f} GB")
    except ImportError:
        print("psutil not available for RAM check")
    
    try:
        import shutil
        total, used, free = shutil.disk_usage("/")
        print(f"Free Disk Space: {free / (1024**3):.2f} GB")
    except:
        print("Disk space check not available")

# Run checks
TREE_METHOD, DEVICE = check_gpu_for_xgboost()
check_system_resources()

# Set random seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
print(f"\n✓ Random seed set to {SEED} for reproducibility")

# ==============================================================================
# SECTION 3: CONFIGURATION AND HYPERPARAMETERS
# ==============================================================================

print("\n" + "=" * 80)
print("SECTION 3: CONFIGURATION AND HYPERPARAMETERS")
print("=" * 80)

class Config:
    """Central configuration class for all hyperparameters and settings."""
    
    # Dataset paths
    GDRIVE_FILE_ID = "16iWDam4flWoyrga-42VlNogohx6nPXYm"
    DATASET_ZIP_NAME = "dataset.zip"
    EXTRACT_PATH = "/content/dataset"
    
    # Data paths (after extraction)
    TRAIN_BENIGN_PATH = "Phishing Website Detection Dataset/Data/Train/benign_Train.txt"
    TRAIN_MALIGN_PATH = "Phishing Website Detection Dataset/Data/Train/malign_Train.txt"
    TEST_BENIGN_PATH = "Phishing Website Detection Dataset/Data/Test/benign_Test.txt"
    TEST_MALIGN_PATH = "Phishing Website Detection Dataset/Data/Test/malign_Test.txt"
    
    # Sampling configuration
    TRAIN_SAMPLE_SIZE = 500000  # 500K samples for training
    TEST_SAMPLE_SIZE = 100000   # 100K samples for testing
    VALIDATION_SPLIT = 0.1     # 10% of training data for validation
    
    # XGBoost hyperparameters (optimized for URL classification)
    XGBOOST_PARAMS = {
        'n_estimators': 500,
        'max_depth': 8,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 5,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'scale_pos_weight': 1,
        'random_state': SEED,
        'n_jobs': -1,
        'eval_metric': 'auc'
    }
    
    # Early stopping
    EARLY_STOPPING_ROUNDS = 50
    
    # Output paths
    MODEL_SAVE_PATH = "/content/phishing_detector_xgboost.json"
    SCALER_SAVE_PATH = "/content/feature_scaler.pkl"
    RESULTS_PATH = "/content/results_xgboost"

# Print configuration
config = Config()
print("\n--- Configuration Summary ---")
print(f"Training samples: {config.TRAIN_SAMPLE_SIZE:,}")
print(f"Test samples: {config.TEST_SAMPLE_SIZE:,}")
print(f"XGBoost estimators: {config.XGBOOST_PARAMS['n_estimators']}")
print(f"Max depth: {config.XGBOOST_PARAMS['max_depth']}")
print(f"Learning rate: {config.XGBOOST_PARAMS['learning_rate']}")
print(f"Tree method: {TREE_METHOD}")
print(f"Device: {DEVICE}")
print("\n✓ Configuration loaded successfully!")


# ==============================================================================
# SECTION 4: DATA DOWNLOAD AND EXTRACTION
# ==============================================================================

print("\n" + "=" * 80)
print("SECTION 4: DATA DOWNLOAD AND EXTRACTION")
print("=" * 80)

def mount_google_drive():
    """Mount Google Drive for Colab environment."""
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✓ Google Drive mounted successfully!")
        return True
    except ImportError:
        print("⚠ Not running in Colab environment")
        return False
    except Exception as e:
        print(f"✗ Error mounting Google Drive: {e}")
        return False

def download_dataset_from_gdrive(file_id, output_path):
    """Download dataset from Google Drive using gdown."""
    import gdown
    
    print(f"\n--- Downloading Dataset ---")
    print(f"File ID: {file_id}")
    
    url = f"https://drive.google.com/uc?id={file_id}"
    
    try:
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / (1024**2)
            print(f"✓ Dataset already downloaded: {output_path} ({file_size:.2f} MB)")
            return True
        
        print("Downloading... (this may take a few minutes)")
        gdown.download(url, output_path, quiet=False, fuzzy=True)
        
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / (1024**2)
            print(f"✓ Download complete! File size: {file_size:.2f} MB")
            return True
        else:
            print("✗ Download failed - file not found")
            return False
            
    except Exception as e:
        print(f"✗ Download error: {e}")
        print("  Please manually download and upload to /content/")
        return False

def extract_dataset(zip_path, extract_to):
    """Extract the downloaded ZIP file."""
    print(f"\n--- Extracting Dataset ---")
    
    try:
        if os.path.exists(extract_to) and os.listdir(extract_to):
            print(f"✓ Dataset already extracted to: {extract_to}")
            return True
        
        os.makedirs(extract_to, exist_ok=True)
        
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        
        print(f"✓ Extraction complete!")
        return True
        
    except zipfile.BadZipFile:
        print("✗ Error: Invalid ZIP file")
        return False
    except Exception as e:
        print(f"✗ Extraction error: {e}")
        return False

# Execute data download and extraction
print("\nStep 1: Mounting Google Drive...")
mount_google_drive()

print("\nStep 2: Downloading dataset...")
zip_path = os.path.join("/content", config.DATASET_ZIP_NAME)
download_success = download_dataset_from_gdrive(config.GDRIVE_FILE_ID, zip_path)

if download_success:
    print("\nStep 3: Extracting dataset...")
    extract_success = extract_dataset(zip_path, config.EXTRACT_PATH)
else:
    print("\n⚠ Please manually download the dataset and upload to /content/")

# ==============================================================================
# SECTION 5: FEATURE ENGINEERING
# ==============================================================================

print("\n" + "=" * 80)
print("SECTION 5: FEATURE ENGINEERING")
print("=" * 80)

class URLFeatureExtractor:
    """
    Extract features from URLs for phishing detection.
    Features are based on research literature and domain knowledge.
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
        """Extract all features from a single URL."""
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
    
    def extract_features_batch(self, urls, show_progress=True):
        """Extract features from a list of URLs."""
        features_list = []
        total = len(urls)
        
        for i, url in enumerate(urls):
            features_list.append(self.extract_features(url))
            
            if show_progress and (i + 1) % 50000 == 0:
                print(f"  Processed {i+1:,}/{total:,} URLs ({(i+1)/total*100:.1f}%)")
        
        return pd.DataFrame(features_list)

# Initialize feature extractor
feature_extractor = URLFeatureExtractor()

# Test feature extraction
print("\n--- Testing Feature Extraction ---")
test_url = "http://secure-login.paypal.com.suspicious-site.xyz/verify?id=12345"
test_features = feature_extractor.extract_features(test_url)
print(f"Test URL: {test_url}")
print(f"Number of features: {len(test_features)}")
print(f"\nSample features:")
for key, value in list(test_features.items())[:10]:
    print(f"  {key}: {value}")

print(f"\n✓ Feature extractor ready with {len(test_features)} features!")


# ==============================================================================
# SECTION 6: DATA LOADING AND PREPROCESSING
# ==============================================================================

print("\n" + "=" * 80)
print("SECTION 6: DATA LOADING AND PREPROCESSING")
print("=" * 80)

class URLDataLoader:
    """Class to handle URL data loading and feature extraction."""
    
    def __init__(self, base_path, config, feature_extractor):
        self.base_path = base_path
        self.config = config
        self.feature_extractor = feature_extractor
        
    def load_urls_from_file(self, file_path, label, sample_size=None):
        """Load URLs from a text file with optional sampling."""
        full_path = os.path.join(self.base_path, file_path)
        
        print(f"Loading: {os.path.basename(file_path)}...")
        
        try:
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                urls = [line.strip() for line in f if line.strip()]
            
            print(f"  Total URLs in file: {len(urls):,}")
            
            if sample_size and sample_size < len(urls):
                urls = random.sample(urls, sample_size)
                print(f"  Sampled: {len(urls):,} URLs")
            
            labels = [label] * len(urls)
            return urls, labels
            
        except FileNotFoundError:
            print(f"  ✗ File not found: {full_path}")
            return [], []
        except Exception as e:
            print(f"  ✗ Error loading file: {e}")
            return [], []
    
    def load_and_preprocess_data(self):
        """Load and preprocess all data with feature extraction."""
        print("\n--- Loading Training Data ---")
        
        train_per_class = self.config.TRAIN_SAMPLE_SIZE // 2 if self.config.TRAIN_SAMPLE_SIZE else None
        test_per_class = self.config.TEST_SAMPLE_SIZE // 2 if self.config.TEST_SAMPLE_SIZE else None
        
        # Load training data
        benign_urls, benign_labels = self.load_urls_from_file(
            self.config.TRAIN_BENIGN_PATH, label=0, sample_size=train_per_class
        )
        malign_urls, malign_labels = self.load_urls_from_file(
            self.config.TRAIN_MALIGN_PATH, label=1, sample_size=train_per_class
        )
        
        train_urls = benign_urls + malign_urls
        train_labels = benign_labels + malign_labels
        
        print(f"\nTotal training URLs: {len(train_urls):,}")
        
        print("\n--- Loading Test Data ---")
        
        test_benign_urls, test_benign_labels = self.load_urls_from_file(
            self.config.TEST_BENIGN_PATH, label=0, sample_size=test_per_class
        )
        test_malign_urls, test_malign_labels = self.load_urls_from_file(
            self.config.TEST_MALIGN_PATH, label=1, sample_size=test_per_class
        )
        
        test_urls = test_benign_urls + test_malign_urls
        test_labels = test_benign_labels + test_malign_labels
        
        print(f"\nTotal test URLs: {len(test_urls):,}")
        
        # Extract features
        print("\n--- Extracting Features (Training Data) ---")
        X_train_full = self.feature_extractor.extract_features_batch(train_urls)
        
        print("\n--- Extracting Features (Test Data) ---")
        X_test = self.feature_extractor.extract_features_batch(test_urls)
        
        y_train_full = np.array(train_labels)
        y_test = np.array(test_labels)
        
        # Handle any NaN or infinite values
        print("\n--- Cleaning Data ---")
        X_train_full = X_train_full.replace([np.inf, -np.inf], np.nan)
        X_test = X_test.replace([np.inf, -np.inf], np.nan)
        X_train_full = X_train_full.fillna(0)
        X_test = X_test.fillna(0)
        
        # Split training into train and validation
        print("\n--- Creating Validation Split ---")
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full,
            test_size=self.config.VALIDATION_SPLIT,
            random_state=SEED,
            stratify=y_train_full
        )
        
        # Shuffle data
        shuffle_idx = np.random.permutation(len(X_train))
        X_train = X_train.iloc[shuffle_idx].reset_index(drop=True)
        y_train = y_train[shuffle_idx]
        
        print(f"Training set: {len(X_train):,} samples")
        print(f"Validation set: {len(X_val):,} samples")
        print(f"Test set: {len(X_test):,} samples")
        print(f"Number of features: {X_train.shape[1]}")
        
        # Store URLs for later analysis
        self.train_urls = train_urls
        self.test_urls = test_urls
        
        return X_train, X_val, X_test, y_train, y_val, y_test

# Load and preprocess data
print("\n⏳ Loading and preprocessing data (this may take several minutes)...")
data_loader = URLDataLoader(config.EXTRACT_PATH, config, feature_extractor)

try:
    X_train, X_val, X_test, y_train, y_val, y_test = data_loader.load_and_preprocess_data()
except FileNotFoundError as e:
    print(f"\n✗ Error: Dataset files not found! {e}")
    raise SystemExit("Dataset loading failed.")
except MemoryError:
    print("\n✗ Error: Out of memory! Try reducing sample sizes.")
    raise SystemExit("Memory error.")

# Data summary
print("\n--- Data Summary ---")
print(f"X_train shape: {X_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"Feature names: {list(X_train.columns[:10])}...")
print("\n✓ Data loading and preprocessing complete!")

# ==============================================================================
# SECTION 7: DATA VISUALIZATION
# ==============================================================================

print("\n" + "=" * 80)
print("SECTION 7: DATA VISUALIZATION")
print("=" * 80)

# Create results directory
os.makedirs(config.RESULTS_PATH, exist_ok=True)
print(f"Results will be saved to: {config.RESULTS_PATH}")

def visualize_class_distribution(y_train, y_val, y_test):
    """Visualize class distribution."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    datasets = [
        (y_train, 'Training Set', axes[0]),
        (y_val, 'Validation Set', axes[1]),
        (y_test, 'Test Set', axes[2])
    ]
    
    colors = ['#2ecc71', '#e74c3c']
    labels = ['Benign', 'Phishing']
    
    for data, title, ax in datasets:
        counts = np.bincount(data)
        bars = ax.bar(labels, counts, color=colors, edgecolor='black', linewidth=1.2)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel('Count', fontsize=12)
        
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                   f'{count:,}', ha='center', va='bottom', fontsize=11)
    
    plt.suptitle('Class Distribution Across Datasets', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_PATH, 'class_distribution.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ Class distribution plot saved!")

def visualize_feature_distributions(X_train, y_train, top_n=6):
    """Visualize distributions of top features."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Select some important features
    important_features = ['url_length', 'num_dots', 'url_entropy', 
                         'num_suspicious_keywords', 'digit_ratio', 'path_depth']
    
    for i, feature in enumerate(important_features):
        if feature in X_train.columns:
            ax = axes[i]
            
            benign_data = X_train[y_train == 0][feature]
            phishing_data = X_train[y_train == 1][feature]
            
            ax.hist(benign_data, bins=50, alpha=0.5, label='Benign', color='#2ecc71', density=True)
            ax.hist(phishing_data, bins=50, alpha=0.5, label='Phishing', color='#e74c3c', density=True)
            ax.set_title(feature, fontsize=12, fontweight='bold')
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.legend()
    
    plt.suptitle('Feature Distributions by Class', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_PATH, 'feature_distributions.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ Feature distributions plot saved!")

def visualize_correlation_matrix(X_train, top_n=15):
    """Visualize correlation matrix of top features."""
    # Select subset of features for readability
    selected_features = X_train.columns[:top_n]
    corr_matrix = X_train[selected_features].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=0.5)
    plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_PATH, 'correlation_matrix.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ Correlation matrix saved!")

# Generate visualizations
print("\n--- Generating Data Visualizations ---")
visualize_class_distribution(y_train, y_val, y_test)
visualize_feature_distributions(X_train, y_train)
visualize_correlation_matrix(X_train)


# ==============================================================================
# SECTION 8: FEATURE SCALING
# ==============================================================================

print("\n" + "=" * 80)
print("SECTION 8: FEATURE SCALING")
print("=" * 80)

# Note: XGBoost doesn't require feature scaling, but it can help with 
# interpretability and some regularization techniques

print("\n--- Feature Scaling (Optional for XGBoost) ---")
print("XGBoost is tree-based and doesn't require scaling.")
print("Keeping original features for better interpretability.")

# Store feature names
feature_names = list(X_train.columns)
print(f"\n✓ {len(feature_names)} features ready for training")

# ==============================================================================
# SECTION 9: MODEL TRAINING
# ==============================================================================

print("\n" + "=" * 80)
print("SECTION 9: MODEL TRAINING")
print("=" * 80)

def train_xgboost_model(X_train, y_train, X_val, y_val, config, tree_method, device):
    """Train XGBoost model with early stopping."""
    
    print("\n--- Training XGBoost Model ---")
    print(f"Training samples: {len(X_train):,}")
    print(f"Validation samples: {len(X_val):,}")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"Tree method: {tree_method}")
    print(f"Device: {device}")
    
    # Update params with tree method and device
    params = config.XGBOOST_PARAMS.copy()
    params['tree_method'] = tree_method
    params['device'] = device
    
    # Remove eval_metric from params (will be passed to fit)
    eval_metric = params.pop('eval_metric', 'auc')
    
    print(f"\nHyperparameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    print("\n" + "-" * 50)
    print("Training started at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("-" * 50)
    
    start_time = time.time()
    
    # Create and train model
    model = xgb.XGBClassifier(**params, early_stopping_rounds=config.EARLY_STOPPING_ROUNDS)
    
    # Train with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=50  # Print every 50 iterations
    )
    
    training_time = time.time() - start_time
    
    print("\n" + "-" * 50)
    print("Training completed at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print(f"Total training time: {training_time/60:.2f} minutes")
    print(f"Best iteration: {model.best_iteration if hasattr(model, 'best_iteration') and model.best_iteration else 'N/A'}")
    print("-" * 50)
    
    return model, training_time

# Train the model
print("\n⏳ Starting XGBoost training...")
try:
    model, training_time = train_xgboost_model(X_train, y_train, X_val, y_val, config, TREE_METHOD, DEVICE)
except Exception as e:
    print(f"\n✗ Training error: {e}")
    raise

# ==============================================================================
# SECTION 10: TRAINING VISUALIZATION
# ==============================================================================

print("\n" + "=" * 80)
print("SECTION 10: TRAINING VISUALIZATION")
print("=" * 80)

def plot_training_history(model):
    """Plot XGBoost training history."""
    
    print("\n--- Plotting Training History ---")
    
    results = model.evals_result()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot training curves
    epochs = len(results['validation_0']['logloss'])
    x_axis = range(0, epochs)
    
    # Loss curve
    axes[0].plot(x_axis, results['validation_0']['logloss'], 'b-', label='Training', linewidth=2)
    axes[0].plot(x_axis, results['validation_1']['logloss'], 'r-', label='Validation', linewidth=2)
    axes[0].set_title('Model Loss (Log Loss)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Log Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # AUC curve (if available)
    if 'auc' in results['validation_0']:
        axes[1].plot(x_axis, results['validation_0']['auc'], 'b-', label='Training', linewidth=2)
        axes[1].plot(x_axis, results['validation_1']['auc'], 'r-', label='Validation', linewidth=2)
        axes[1].set_title('Model AUC', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('AUC')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('XGBoost Training History', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_PATH, 'training_history.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✓ Training history plot saved!")

# Plot training history
plot_training_history(model)

# ==============================================================================
# SECTION 11: MODEL EVALUATION
# ==============================================================================

print("\n" + "=" * 80)
print("SECTION 11: MODEL EVALUATION")
print("=" * 80)

def evaluate_model(model, X_test, y_test, threshold=0.5):
    """Comprehensive model evaluation."""
    
    print("\n--- Evaluating Model on Test Set ---")
    print(f"Test samples: {len(X_test):,}")
    
    # Get predictions
    print("Generating predictions...")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n{'='*40}")
    print(f"{'METRIC':<20} {'VALUE':>15}")
    print(f"{'='*40}")
    print(f"{'Accuracy':<20} {accuracy:>15.4f}")
    print(f"{'Precision':<20} {precision:>15.4f}")
    print(f"{'Recall':<20} {recall:>15.4f}")
    print(f"{'F1-Score':<20} {f1:>15.4f}")
    print(f"{'ROC-AUC':<20} {roc_auc:>15.4f}")
    print(f"{'='*40}")
    
    print("\n--- Detailed Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=['Benign', 'Phishing']))
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    return results

def plot_confusion_matrix(y_test, y_pred):
    """Plot confusion matrix."""
    
    print("\n--- Plotting Confusion Matrix ---")
    
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Benign', 'Phishing'],
                yticklabels=['Benign', 'Phishing'],
                annot_kws={'size': 14})
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            percentage = cm[i, j] / total * 100
            plt.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)',
                    ha='center', va='center', fontsize=10, color='gray')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_PATH, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    tn, fp, fn, tp = cm.ravel()
    print(f"\nConfusion Matrix Details:")
    print(f"  True Negatives: {tn:,}")
    print(f"  False Positives: {fp:,}")
    print(f"  False Negatives: {fn:,}")
    print(f"  True Positives: {tp:,}")
    
    print("\n✓ Confusion matrix saved!")

def plot_roc_curve(y_test, y_pred_proba):
    """Plot ROC curve."""
    
    print("\n--- Plotting ROC Curve ---")
    
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='#3498db', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Classifier')
    plt.fill_between(fpr, tpr, alpha=0.3, color='#3498db')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_PATH, 'roc_curve.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print(f"\nOptimal threshold: {optimal_threshold:.4f}")
    
    print("\n✓ ROC curve saved!")
    return optimal_threshold

def plot_precision_recall_curve(y_test, y_pred_proba):
    """Plot Precision-Recall curve."""
    
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    print("\n--- Plotting Precision-Recall Curve ---")
    
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall_vals, precision_vals, color='#e74c3c', lw=2,
             label=f'PR Curve (AP = {avg_precision:.4f})')
    plt.fill_between(recall_vals, precision_vals, alpha=0.3, color='#e74c3c')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    plt.legend(loc='lower left', fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_PATH, 'precision_recall_curve.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n✓ Precision-Recall curve saved!")

# Evaluate model
results = evaluate_model(model, X_test, y_test)

# Plot evaluation visualizations
plot_confusion_matrix(y_test, results['y_pred'])
optimal_threshold = plot_roc_curve(y_test, results['y_pred_proba'])
plot_precision_recall_curve(y_test, results['y_pred_proba'])


# ==============================================================================
# SECTION 12: FEATURE IMPORTANCE ANALYSIS
# ==============================================================================

print("\n" + "=" * 80)
print("SECTION 12: FEATURE IMPORTANCE ANALYSIS")
print("=" * 80)

def plot_feature_importance(model, feature_names, top_n=20):
    """Plot feature importance from XGBoost."""
    
    print("\n--- Feature Importance Analysis ---")
    
    # Get feature importance
    importance = model.feature_importances_
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    # Plot top N features
    plt.figure(figsize=(12, 8))
    top_features = importance_df.head(top_n)
    
    bars = plt.barh(range(len(top_features)), top_features['importance'].values, color='#3498db')
    plt.yticks(range(len(top_features)), top_features['feature'].values)
    plt.xlabel('Feature Importance (Gain)', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title(f'Top {top_n} Most Important Features', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, top_features['importance'].values)):
        plt.text(val + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{val:.4f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_PATH, 'feature_importance.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nTop {top_n} Most Important Features:")
    for i, row in importance_df.head(top_n).iterrows():
        print(f"  {importance_df.index.get_loc(i)+1}. {row['feature']}: {row['importance']:.4f}")
    
    print("\n✓ Feature importance plot saved!")
    
    return importance_df

# Plot feature importance
importance_df = plot_feature_importance(model, feature_names)

# ==============================================================================
# SECTION 13: MODEL EXPLAINABILITY (SHAP)
# ==============================================================================

print("\n" + "=" * 80)
print("SECTION 13: MODEL EXPLAINABILITY (SHAP)")
print("=" * 80)

def explain_model_with_shap(model, X_train, X_test, feature_names, num_samples=1000):
    """Use SHAP to explain model predictions."""
    
    print("\n--- SHAP Explainability Analysis ---")
    print(f"Using {num_samples} samples for SHAP analysis")
    
    try:
        # Sample data for SHAP
        sample_indices = np.random.choice(len(X_test), min(num_samples, len(X_test)), replace=False)
        X_sample = X_test.iloc[sample_indices]
        
        print("Creating SHAP explainer...")
        explainer = shap.TreeExplainer(model)
        
        print("Calculating SHAP values...")
        shap_values = explainer.shap_values(X_sample)
        
        # Summary plot
        print("\n--- SHAP Summary Plot ---")
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, 
                         show=False, max_display=20)
        plt.title('SHAP Feature Importance', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(config.RESULTS_PATH, 'shap_summary.png'), dpi=150, bbox_inches='tight')
        plt.show()
        
        # Bar plot
        print("\n--- SHAP Bar Plot ---")
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                         plot_type='bar', show=False, max_display=20)
        plt.title('SHAP Feature Importance (Bar)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(config.RESULTS_PATH, 'shap_bar.png'), dpi=150, bbox_inches='tight')
        plt.show()
        
        print("\n✓ SHAP analysis complete!")
        
        return explainer, shap_values
        
    except Exception as e:
        print(f"\n⚠ SHAP analysis error: {e}")
        return None, None

# Run SHAP analysis
try:
    explainer, shap_values = explain_model_with_shap(model, X_train, X_test, feature_names)
except Exception as e:
    print(f"⚠ SHAP analysis skipped: {e}")
    explainer, shap_values = None, None

# ==============================================================================
# SECTION 14: MODEL SAVING AND LOADING
# ==============================================================================

print("\n" + "=" * 80)
print("SECTION 14: MODEL SAVING AND LOADING")
print("=" * 80)

def save_model(model, config, feature_names):
    """Save the trained model and feature names."""
    
    print("\n--- Saving Model ---")
    
    # Save XGBoost model
    model.save_model(config.MODEL_SAVE_PATH)
    print(f"✓ Model saved to: {config.MODEL_SAVE_PATH}")
    
    model_size = os.path.getsize(config.MODEL_SAVE_PATH) / (1024**2)
    print(f"  Model size: {model_size:.2f} MB")
    
    # Save feature names
    import pickle
    feature_path = config.MODEL_SAVE_PATH.replace('.json', '_features.pkl')
    with open(feature_path, 'wb') as f:
        pickle.dump(feature_names, f)
    print(f"✓ Feature names saved to: {feature_path}")
    
    return config.MODEL_SAVE_PATH

def load_and_verify_model(model_path, X_test, y_test):
    """Load and verify the saved model."""
    
    print("\n--- Loading and Verifying Model ---")
    
    try:
        loaded_model = xgb.XGBClassifier()
        loaded_model.load_model(model_path)
        print(f"✓ Model loaded from: {model_path}")
        
        # Verify predictions
        sample_size = min(1000, len(X_test))
        X_sample = X_test.iloc[:sample_size]
        y_sample = y_test[:sample_size]
        
        predictions = loaded_model.predict(X_sample)
        accuracy = accuracy_score(y_sample, predictions)
        print(f"✓ Verification accuracy on {sample_size} samples: {accuracy:.4f}")
        
        print("\n✓ Model loading and verification successful!")
        return loaded_model
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None

# Save the model
saved_model_path = save_model(model, config, feature_names)

# Load and verify
loaded_model = load_and_verify_model(saved_model_path, X_test, y_test)

# ==============================================================================
# SECTION 15: PREDICTION FUNCTION FOR NEW URLs
# ==============================================================================

print("\n" + "=" * 80)
print("SECTION 15: PREDICTION FUNCTION FOR NEW URLs")
print("=" * 80)

class PhishingURLDetector:
    """Class for making predictions on new URLs using XGBoost."""
    
    def __init__(self, model, feature_extractor):
        self.model = model
        self.feature_extractor = feature_extractor
    
    def predict(self, url):
        """Predict if a URL is phishing or benign."""
        # Extract features
        features = self.feature_extractor.extract_features(url)
        X = pd.DataFrame([features])
        
        # Handle NaN/inf
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Predict
        probability = self.model.predict_proba(X)[0][1]
        is_phishing = probability >= 0.5
        confidence = probability if is_phishing else 1 - probability
        
        return {
            'url': url,
            'is_phishing': is_phishing,
            'probability': float(probability),
            'confidence': float(confidence),
            'label': 'PHISHING' if is_phishing else 'BENIGN'
        }
    
    def predict_batch(self, urls):
        """Predict for multiple URLs."""
        return [self.predict(url) for url in urls]
    
    def display_prediction(self, url):
        """Display prediction with formatting."""
        result = self.predict(url)
        
        print(f"\n{'='*60}")
        print(f"URL: {url[:50]}..." if len(url) > 50 else f"URL: {url}")
        print(f"{'='*60}")
        
        if result['is_phishing']:
            print(f"⚠️  PREDICTION: {result['label']}")
            print(f"🔴 Phishing Probability: {result['probability']:.2%}")
        else:
            print(f"✅ PREDICTION: {result['label']}")
            print(f"🟢 Benign Probability: {1-result['probability']:.2%}")
        
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"{'='*60}")
        
        return result

# Create detector instance
detector = PhishingURLDetector(model, feature_extractor)

# Test with example URLs
print("\n--- Testing Prediction Function ---")

test_urls_examples = [
    "google.com",
    "facebook.com/login",
    "secure-banking-login.suspicious-site.com/verify",
    "paypal.com.fake-domain.net/signin",
    "amazon.com",
    "free-iphone-winner.xyz/claim-prize",
    "github.com/user/repo",
    "login-verify-account.com/secure/bank"
]

print("\nPredictions for sample URLs:")
for url in test_urls_examples:
    detector.display_prediction(url)


# ==============================================================================
# SECTION 16: FINAL SUMMARY AND RESULTS EXPORT
# ==============================================================================

print("\n" + "=" * 80)
print("SECTION 16: FINAL SUMMARY AND RESULTS EXPORT")
print("=" * 80)

def generate_final_report(results, training_time, config, importance_df):
    """Generate comprehensive final report."""
    
    print("\n" + "=" * 80)
    print("                    FINAL TRAINING REPORT")
    print("=" * 80)
    
    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         PHISHING URL DETECTION MODEL                         ║
║                              XGBoost Classifier                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  DATASET INFORMATION                                                         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Training Samples:     {config.TRAIN_SAMPLE_SIZE:>10,}                                          ║
║  Validation Split:     {config.VALIDATION_SPLIT*100:>10.0f}%                                          ║
║  Test Samples:         {config.TEST_SAMPLE_SIZE:>10,}                                          ║
║  Number of Features:   {len(feature_names):>10}                                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  MODEL CONFIGURATION                                                         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Algorithm:            {'XGBoost':>10}                                          ║
║  N Estimators:         {config.XGBOOST_PARAMS['n_estimators']:>10}                                          ║
║  Max Depth:            {config.XGBOOST_PARAMS['max_depth']:>10}                                          ║
║  Learning Rate:        {config.XGBOOST_PARAMS['learning_rate']:>10}                                        ║
║  Tree Method:          {TREE_METHOD:>10}                                          ║
║  Device:               {DEVICE:>10}                                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  TRAINING INFORMATION                                                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Training Time:        {training_time/60:>10.2f} min                                     ║
║  Best Iteration:       {model.best_iteration if hasattr(model, 'best_iteration') and model.best_iteration else 500:>10}                                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  EVALUATION METRICS (TEST SET)                                               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Accuracy:             {results['accuracy']:>10.4f}                                          ║
║  Precision:            {results['precision']:>10.4f}                                          ║
║  Recall:               {results['recall']:>10.4f}                                          ║
║  F1-Score:             {results['f1_score']:>10.4f}                                          ║
║  ROC-AUC:              {results['roc_auc']:>10.4f}                                          ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
    
    # Top features
    print("\n--- Top 10 Most Important Features ---")
    for i, row in importance_df.head(10).iterrows():
        idx = importance_df.index.get_loc(i) + 1
        print(f"  {idx:2}. {row['feature']:<30} {row['importance']:.4f}")
    
    # Save results to JSON
    import json
    results_summary = {
        'dataset': {
            'train_samples': config.TRAIN_SAMPLE_SIZE,
            'test_samples': config.TEST_SAMPLE_SIZE,
            'num_features': len(feature_names)
        },
        'model': {
            'algorithm': 'XGBoost',
            'n_estimators': config.XGBOOST_PARAMS['n_estimators'],
            'max_depth': config.XGBOOST_PARAMS['max_depth'],
            'learning_rate': config.XGBOOST_PARAMS['learning_rate'],
            'tree_method': TREE_METHOD,
            'device': DEVICE,
            'best_iteration': int(model.best_iteration) if hasattr(model, 'best_iteration') and model.best_iteration else config.XGBOOST_PARAMS['n_estimators']
        },
        'training': {
            'training_time_minutes': training_time / 60
        },
        'metrics': {
            'accuracy': results['accuracy'],
            'precision': results['precision'],
            'recall': results['recall'],
            'f1_score': results['f1_score'],
            'roc_auc': results['roc_auc']
        },
        'top_features': importance_df.head(20).to_dict('records')
    }
    
    results_json_path = os.path.join(config.RESULTS_PATH, 'results_summary.json')
    with open(results_json_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"\n✓ Results summary saved to: {results_json_path}")
    
    return results_summary

def list_saved_files(config):
    """List all saved files."""
    
    print("\n--- Saved Files ---")
    
    files_to_check = [
        (config.MODEL_SAVE_PATH, "Trained Model"),
        (config.MODEL_SAVE_PATH.replace('.json', '_features.pkl'), "Feature Names"),
        (os.path.join(config.RESULTS_PATH, 'results_summary.json'), "Results Summary"),
        (os.path.join(config.RESULTS_PATH, 'training_history.png'), "Training History Plot"),
        (os.path.join(config.RESULTS_PATH, 'confusion_matrix.png'), "Confusion Matrix"),
        (os.path.join(config.RESULTS_PATH, 'roc_curve.png'), "ROC Curve"),
        (os.path.join(config.RESULTS_PATH, 'precision_recall_curve.png'), "PR Curve"),
        (os.path.join(config.RESULTS_PATH, 'feature_importance.png'), "Feature Importance"),
        (os.path.join(config.RESULTS_PATH, 'shap_summary.png'), "SHAP Summary"),
        (os.path.join(config.RESULTS_PATH, 'class_distribution.png'), "Class Distribution"),
    ]
    
    for file_path, description in files_to_check:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path) / 1024
            print(f"  ✓ {description}: {file_path} ({size:.1f} KB)")
        else:
            print(f"  ✗ {description}: Not found")

# Generate final report
final_results = generate_final_report(results, training_time, config, importance_df)

# List saved files
list_saved_files(config)

# ==============================================================================
# SECTION 17: DOWNLOAD FILES (FOR COLAB)
# ==============================================================================

print("\n" + "=" * 80)
print("SECTION 17: DOWNLOAD FILES")
print("=" * 80)

def create_download_zip(config):
    """Create ZIP file with all results."""
    
    print("\n--- Creating Download Package ---")
    
    zip_filename = "/content/phishing_detection_xgboost_results.zip"
    
    try:
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add model
            if os.path.exists(config.MODEL_SAVE_PATH):
                zipf.write(config.MODEL_SAVE_PATH, os.path.basename(config.MODEL_SAVE_PATH))
            
            # Add feature names
            feature_path = config.MODEL_SAVE_PATH.replace('.json', '_features.pkl')
            if os.path.exists(feature_path):
                zipf.write(feature_path, os.path.basename(feature_path))
            
            # Add all results
            if os.path.exists(config.RESULTS_PATH):
                for root, dirs, files in os.walk(config.RESULTS_PATH):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.join('results', file)
                        zipf.write(file_path, arcname)
        
        zip_size = os.path.getsize(zip_filename) / (1024**2)
        print(f"✓ Download package created: {zip_filename} ({zip_size:.2f} MB)")
        
        try:
            from google.colab import files
            print("\n📥 Click below to download the results:")
            files.download(zip_filename)
        except ImportError:
            print(f"\n📁 Download the file from: {zip_filename}")
        
        return zip_filename
        
    except Exception as e:
        print(f"✗ Error creating download package: {e}")
        return None

# Create download package
download_zip = create_download_zip(config)

# ==============================================================================
# FINAL MESSAGE
# ==============================================================================

print("\n" + "=" * 80)
print("                         TRAINING COMPLETE!")
print("=" * 80)
print(f"""
🎉 Congratulations! Your XGBoost phishing URL detection model is ready!

📊 Model Performance:
   • Accuracy:  {results['accuracy']:.2%}
   • Precision: {results['precision']:.2%}
   • Recall:    {results['recall']:.2%}
   • F1-Score:  {results['f1_score']:.2%}
   • ROC-AUC:   {results['roc_auc']:.4f}

📁 Files Saved:
   • Model: {config.MODEL_SAVE_PATH}
   • Results: {config.RESULTS_PATH}/
   • Download: /content/phishing_detection_xgboost_results.zip

🔮 To use the model for predictions:
   detector = PhishingURLDetector(model, feature_extractor)
   result = detector.predict("suspicious-url.com")

📝 For your FYP report, all visualizations are saved in the results folder.

🚀 XGBoost Advantages:
   • Fast training and inference
   • Interpretable feature importance
   • No GPU required (but benefits from it)
   • Excellent for tabular/structured data

Good luck with your Final Year Project! 🎓
""")
print("=" * 80)
