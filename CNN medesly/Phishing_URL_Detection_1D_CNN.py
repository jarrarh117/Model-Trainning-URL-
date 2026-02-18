"""
================================================================================
PHISHING URL DETECTION USING 1D-CNN
================================================================================
Author: FYP Project
Date: January 2026
Description: Complete end-to-end pipeline for detecting phishing URLs using
             1D Convolutional Neural Network with character-level embeddings.
             
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

# Install required packages (if not already installed)
import subprocess
import sys

def install_package(package):
    """Install a package using pip if not already installed."""
    try:
        __import__(package.split('[')[0])
        print(f"✓ {package} already installed")
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
        print(f"✓ {package} installed successfully")

# Install necessary packages
packages_to_install = ['gdown', 'shap', 'scikit-learn', 'tensorflow', 'seaborn', 'pandas']
for pkg in packages_to_install:
    install_package(pkg)

# Core imports
import os
import zipfile
import warnings
import random
import time
from datetime import datetime

# Data handling
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Explainability
import shap

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("\n✓ All imports successful!")

# ==============================================================================
# SECTION 2: GPU CONFIGURATION AND SYSTEM CHECK
# ==============================================================================

print("\n" + "=" * 80)
print("SECTION 2: GPU CONFIGURATION AND SYSTEM CHECK")
print("=" * 80)

def check_gpu_availability():
    """Check and configure GPU for training."""
    print("\n--- GPU Status ---")
    
    # Check TensorFlow version
    print(f"TensorFlow Version: {tf.__version__}")
    
    # Check for GPU
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        print(f"✓ GPU Available: {len(gpus)} GPU(s) detected")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
        
        # Configure GPU memory growth to avoid OOM errors
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✓ GPU memory growth enabled")
        except RuntimeError as e:
            print(f"⚠ GPU memory config error: {e}")
        
        device = '/GPU:0'
    else:
        print("⚠ No GPU detected. Using CPU (training will be slower)")
        device = '/CPU:0'
    
    return device

def check_system_resources():
    """Check available system resources."""
    print("\n--- System Resources ---")
    
    # Check RAM (Colab specific)
    try:
        import psutil
        ram = psutil.virtual_memory()
        print(f"Total RAM: {ram.total / (1024**3):.2f} GB")
        print(f"Available RAM: {ram.available / (1024**3):.2f} GB")
    except ImportError:
        print("psutil not available for RAM check (install with: pip install psutil)")
    
    # Check disk space (Linux/Colab)
    try:
        disk = os.statvfs('/')
        free_space = (disk.f_bavail * disk.f_frsize) / (1024**3)
        print(f"Free Disk Space: {free_space:.2f} GB")
    except AttributeError:
        # Windows doesn't have statvfs
        try:
            import shutil
            total, used, free = shutil.disk_usage("/")
            print(f"Free Disk Space: {free / (1024**3):.2f} GB")
        except:
            print("Disk space check not available")

# Run checks
DEVICE = check_gpu_availability()
check_system_resources()

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
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
    
    # Sampling configuration (for Colab memory constraints)
    # Set to None to use full dataset (requires high RAM)
    TRAIN_SAMPLE_SIZE = 500000  # 500K samples for training (250K each class)
    TEST_SAMPLE_SIZE = 100000   # 100K samples for testing (50K each class)
    VALIDATION_SPLIT = 0.1     # 10% of training data for validation
    
    # Character-level encoding
    MAX_URL_LENGTH = 200       # Maximum URL length (characters)
    VOCAB_SIZE = 128           # ASCII characters (0-127)
    
    # Model architecture
    EMBEDDING_DIM = 64         # Character embedding dimension
    CONV_FILTERS = [128, 128, 64]  # Filters for each Conv1D layer
    KERNEL_SIZES = [3, 5, 7]   # Kernel sizes for each Conv1D layer
    DENSE_UNITS = [128, 64]    # Dense layer units
    DROPOUT_RATE = 0.5         # Dropout rate
    
    # Training configuration
    BATCH_SIZE = 256           # Batch size
    EPOCHS = 20                # Maximum epochs
    LEARNING_RATE = 0.001      # Initial learning rate
    EARLY_STOPPING_PATIENCE = 5  # Early stopping patience
    LR_REDUCE_PATIENCE = 3     # Learning rate reduction patience
    LR_REDUCE_FACTOR = 0.5     # Learning rate reduction factor
    
    # Output paths
    MODEL_SAVE_PATH = "/content/phishing_detector_model.keras"
    HISTORY_SAVE_PATH = "/content/training_history.npy"
    RESULTS_PATH = "/content/results"

# Print configuration
config = Config()
print("\n--- Configuration Summary ---")
print(f"Training samples: {config.TRAIN_SAMPLE_SIZE:,} (per class: {config.TRAIN_SAMPLE_SIZE//2:,})")
print(f"Test samples: {config.TEST_SAMPLE_SIZE:,}")
print(f"Max URL length: {config.MAX_URL_LENGTH}")
print(f"Embedding dimension: {config.EMBEDDING_DIM}")
print(f"Batch size: {config.BATCH_SIZE}")
print(f"Max epochs: {config.EPOCHS}")
print(f"Learning rate: {config.LEARNING_RATE}")
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
        # Check if file already exists
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
            print("  Try manually downloading from Google Drive and uploading to Colab")
            return False
            
    except Exception as e:
        print(f"✗ Download error: {e}")
        print("  Attempting alternative download method...")
        try:
            # Alternative: direct download with confirmation bypass
            gdown.download(url, output_path, quiet=False, fuzzy=True, use_cookies=False)
            if os.path.exists(output_path):
                return True
        except:
            pass
        print("  Please manually download the dataset and upload to /content/")
        return False

def extract_dataset(zip_path, extract_to):
    """Extract the downloaded ZIP file."""
    print(f"\n--- Extracting Dataset ---")
    
    try:
        # Check if already extracted
        if os.path.exists(extract_to) and os.listdir(extract_to):
            print(f"✓ Dataset already extracted to: {extract_to}")
            return True
        
        # Create extraction directory
        os.makedirs(extract_to, exist_ok=True)
        
        print(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        
        print(f"✓ Extraction complete!")
        
        # List extracted contents
        print("\nExtracted contents:")
        for root, dirs, files in os.walk(extract_to):
            level = root.replace(extract_to, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path) / (1024**2)
                print(f"{subindent}{file} ({file_size:.2f} MB)")
        
        return True
        
    except zipfile.BadZipFile:
        print("✗ Error: Invalid ZIP file")
        return False
    except Exception as e:
        print(f"✗ Extraction error: {e}")
        return False

def verify_dataset_files(base_path):
    """Verify all required dataset files exist."""
    print(f"\n--- Verifying Dataset Files ---")
    
    required_files = [
        config.TRAIN_BENIGN_PATH,
        config.TRAIN_MALIGN_PATH,
        config.TEST_BENIGN_PATH,
        config.TEST_MALIGN_PATH
    ]
    
    all_found = True
    for file_path in required_files:
        full_path = os.path.join(base_path, file_path)
        if os.path.exists(full_path):
            file_size = os.path.getsize(full_path) / (1024**2)
            # Count lines efficiently without loading entire file
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                line_count = sum(1 for _ in f)
            print(f"✓ {os.path.basename(file_path)}: {file_size:.2f} MB, {line_count:,} URLs")
        else:
            print(f"✗ Missing: {file_path}")
            all_found = False
    
    return all_found

# Execute data download and extraction
print("\nStep 1: Mounting Google Drive...")
mount_google_drive()

print("\nStep 2: Downloading dataset...")
zip_path = os.path.join("/content", config.DATASET_ZIP_NAME)
download_success = download_dataset_from_gdrive(config.GDRIVE_FILE_ID, zip_path)

if download_success:
    print("\nStep 3: Extracting dataset...")
    extract_success = extract_dataset(zip_path, config.EXTRACT_PATH)
    
    if extract_success:
        print("\nStep 4: Verifying dataset files...")
        files_verified = verify_dataset_files(config.EXTRACT_PATH)
        if not files_verified:
            print("\n⚠ Some dataset files are missing. Please check the extraction.")
else:
    print("\n" + "="*60)
    print("⚠ MANUAL DOWNLOAD REQUIRED")
    print("="*60)
    print("""
If automatic download fails, please:
1. Download the dataset manually from Google Drive
2. Upload the ZIP file to Colab using the file browser (left sidebar)
3. Rename it to 'dataset.zip' and place it in /content/
4. Re-run this cell
    """)

# ==============================================================================
# SECTION 5: DATA LOADING AND PREPROCESSING
# ==============================================================================

print("\n" + "=" * 80)
print("SECTION 5: DATA LOADING AND PREPROCESSING")
print("=" * 80)

class URLDataLoader:
    """Class to handle URL data loading and preprocessing."""
    
    def __init__(self, base_path, config):
        self.base_path = base_path
        self.config = config
        self.char_to_idx = {chr(i): i for i in range(config.VOCAB_SIZE)}
        
    def load_urls_from_file(self, file_path, label, sample_size=None):
        """Load URLs from a text file with optional sampling."""
        full_path = os.path.join(self.base_path, file_path)
        
        print(f"Loading: {os.path.basename(file_path)}...")
        
        try:
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                urls = [line.strip() for line in f if line.strip()]
            
            print(f"  Total URLs in file: {len(urls):,}")
            
            # Sample if needed
            if sample_size and sample_size < len(urls):
                urls = random.sample(urls, sample_size)
                print(f"  Sampled: {len(urls):,} URLs")
            
            # Create labels
            labels = [label] * len(urls)
            
            return urls, labels
            
        except FileNotFoundError:
            print(f"  ✗ File not found: {full_path}")
            return [], []
        except Exception as e:
            print(f"  ✗ Error loading file: {e}")
            return [], []
    
    def url_to_sequence(self, url):
        """Convert URL string to sequence of character indices."""
        sequence = []
        for char in url[:self.config.MAX_URL_LENGTH]:
            # Get ASCII value, use 0 for unknown characters
            idx = ord(char) if ord(char) < self.config.VOCAB_SIZE else 0
            sequence.append(idx)
        return sequence
    
    def preprocess_urls(self, urls):
        """Convert list of URLs to padded sequences."""
        print(f"Converting {len(urls):,} URLs to sequences...")
        
        # Use batch processing for memory efficiency
        batch_size = 50000
        all_sequences = []
        
        for i in range(0, len(urls), batch_size):
            batch = urls[i:i+batch_size]
            sequences = [self.url_to_sequence(url) for url in batch]
            all_sequences.extend(sequences)
            if (i + batch_size) % 100000 == 0:
                print(f"  Processed {min(i+batch_size, len(urls)):,}/{len(urls):,} URLs...")
        
        # Pad sequences to fixed length
        padded = pad_sequences(
            all_sequences,
            maxlen=self.config.MAX_URL_LENGTH,
            padding='post',
            truncating='post',
            value=0
        )
        
        print(f"  Sequence shape: {padded.shape}")
        return padded
    
    def load_and_preprocess_data(self):
        """Load and preprocess all data."""
        print("\n--- Loading Training Data ---")
        
        # Calculate per-class sample size
        train_per_class = self.config.TRAIN_SAMPLE_SIZE // 2 if self.config.TRAIN_SAMPLE_SIZE else None
        test_per_class = self.config.TEST_SAMPLE_SIZE // 2 if self.config.TEST_SAMPLE_SIZE else None
        
        # Load training data
        benign_urls, benign_labels = self.load_urls_from_file(
            self.config.TRAIN_BENIGN_PATH, label=0, sample_size=train_per_class
        )
        malign_urls, malign_labels = self.load_urls_from_file(
            self.config.TRAIN_MALIGN_PATH, label=1, sample_size=train_per_class
        )
        
        # Combine training data
        train_urls = benign_urls + malign_urls
        train_labels = benign_labels + malign_labels
        
        print(f"\nTotal training URLs: {len(train_urls):,}")
        print(f"  Benign: {len(benign_urls):,}")
        print(f"  Malign: {len(malign_urls):,}")
        
        print("\n--- Loading Test Data ---")
        
        # Load test data
        test_benign_urls, test_benign_labels = self.load_urls_from_file(
            self.config.TEST_BENIGN_PATH, label=0, sample_size=test_per_class
        )
        test_malign_urls, test_malign_labels = self.load_urls_from_file(
            self.config.TEST_MALIGN_PATH, label=1, sample_size=test_per_class
        )
        
        # Combine test data
        test_urls = test_benign_urls + test_malign_urls
        test_labels = test_benign_labels + test_malign_labels
        
        print(f"\nTotal test URLs: {len(test_urls):,}")
        print(f"  Benign: {len(test_benign_urls):,}")
        print(f"  Malign: {len(test_malign_urls):,}")
        
        # Preprocess URLs
        print("\n--- Preprocessing URLs ---")
        X_train_full = self.preprocess_urls(train_urls)
        X_test = self.preprocess_urls(test_urls)
        
        y_train_full = np.array(train_labels)
        y_test = np.array(test_labels)
        
        # Split training into train and validation
        print("\n--- Creating Validation Split ---")
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full,
            test_size=self.config.VALIDATION_SPLIT,
            random_state=SEED,
            stratify=y_train_full
        )
        
        # Shuffle training data
        print("Shuffling training data...")
        shuffle_idx = np.random.permutation(len(X_train))
        X_train = X_train[shuffle_idx]
        y_train = y_train[shuffle_idx]
        
        # Shuffle test data
        shuffle_idx_test = np.random.permutation(len(X_test))
        X_test = X_test[shuffle_idx_test]
        y_test = y_test[shuffle_idx_test]
        
        print(f"Training set: {X_train.shape[0]:,} samples")
        print(f"Validation set: {X_val.shape[0]:,} samples")
        print(f"Test set: {X_test.shape[0]:,} samples")
        
        # Store original URLs for later analysis
        self.train_urls = train_urls
        self.test_urls = test_urls
        
        return X_train, X_val, X_test, y_train, y_val, y_test

# Load and preprocess data
print("\n⏳ Loading and preprocessing data (this may take a few minutes)...")
data_loader = URLDataLoader(config.EXTRACT_PATH, config)

try:
    X_train, X_val, X_test, y_train, y_val, y_test = data_loader.load_and_preprocess_data()
except FileNotFoundError as e:
    print(f"\n✗ Error: Dataset files not found!")
    print(f"  {e}")
    print("\nPlease ensure the dataset is properly extracted to:", config.EXTRACT_PATH)
    raise SystemExit("Dataset loading failed. Please fix the issue and restart.")
except MemoryError:
    print("\n✗ Error: Out of memory!")
    print("  Try reducing TRAIN_SAMPLE_SIZE and TEST_SAMPLE_SIZE in the Config class.")
    raise SystemExit("Memory error. Please reduce sample sizes and restart.")

# Data summary
print("\n--- Data Summary ---")
print(f"X_train shape: {X_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train distribution: {np.bincount(y_train)}")
print(f"y_val distribution: {np.bincount(y_val)}")
print(f"y_test distribution: {np.bincount(y_test)}")
print("\n✓ Data loading and preprocessing complete!")


# ==============================================================================
# SECTION 6: DATA VISUALIZATION
# ==============================================================================

print("\n" + "=" * 80)
print("SECTION 6: DATA VISUALIZATION")
print("=" * 80)

def visualize_data_distribution(y_train, y_val, y_test):
    """Visualize the distribution of classes in train/val/test sets."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    datasets = [
        (y_train, 'Training Set', axes[0]),
        (y_val, 'Validation Set', axes[1]),
        (y_test, 'Test Set', axes[2])
    ]
    
    colors = ['#2ecc71', '#e74c3c']  # Green for benign, Red for malign
    labels = ['Benign', 'Phishing']
    
    for data, title, ax in datasets:
        counts = np.bincount(data)
        bars = ax.bar(labels, counts, color=colors, edgecolor='black', linewidth=1.2)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel('Count', fontsize=12)
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                   f'{count:,}', ha='center', va='bottom', fontsize=11)
    
    plt.suptitle('Class Distribution Across Datasets', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_PATH, 'class_distribution.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ Class distribution plot saved!")

def visualize_url_lengths(train_urls, test_urls, max_length=200):
    """Visualize URL length distribution."""
    
    train_lengths = [min(len(url), max_length) for url in train_urls]
    test_lengths = [min(len(url), max_length) for url in test_urls]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Training URL lengths
    axes[0].hist(train_lengths, bins=50, color='#3498db', edgecolor='black', alpha=0.7)
    axes[0].axvline(np.mean(train_lengths), color='red', linestyle='--', label=f'Mean: {np.mean(train_lengths):.1f}')
    axes[0].axvline(np.median(train_lengths), color='green', linestyle='--', label=f'Median: {np.median(train_lengths):.1f}')
    axes[0].set_title('Training URL Length Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('URL Length (characters)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].legend()
    
    # Test URL lengths
    axes[1].hist(test_lengths, bins=50, color='#9b59b6', edgecolor='black', alpha=0.7)
    axes[1].axvline(np.mean(test_lengths), color='red', linestyle='--', label=f'Mean: {np.mean(test_lengths):.1f}')
    axes[1].axvline(np.median(test_lengths), color='green', linestyle='--', label=f'Median: {np.median(test_lengths):.1f}')
    axes[1].set_title('Test URL Length Distribution', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('URL Length (characters)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_PATH, 'url_length_distribution.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ URL length distribution plot saved!")

# Create results directory
os.makedirs(config.RESULTS_PATH, exist_ok=True)
print(f"Results will be saved to: {config.RESULTS_PATH}")

# Generate visualizations
print("\n--- Generating Data Visualizations ---")
visualize_data_distribution(y_train, y_val, y_test)
visualize_url_lengths(data_loader.train_urls, data_loader.test_urls)

# ==============================================================================
# SECTION 7: MODEL ARCHITECTURE
# ==============================================================================

print("\n" + "=" * 80)
print("SECTION 7: MODEL ARCHITECTURE")
print("=" * 80)

def build_1d_cnn_model(config):
    """
    Build a 1D CNN model for URL classification.
    
    Architecture:
    - Embedding layer for character-level representation
    - Multiple parallel Conv1D branches with different kernel sizes
    - Global Max Pooling
    - Dense layers with dropout
    - Sigmoid output for binary classification
    """
    
    print("\n--- Building 1D-CNN Model ---")
    
    # Input layer
    input_layer = layers.Input(shape=(config.MAX_URL_LENGTH,), name='url_input')
    
    # Embedding layer - converts character indices to dense vectors
    embedding = layers.Embedding(
        input_dim=config.VOCAB_SIZE,
        output_dim=config.EMBEDDING_DIM,
        input_length=config.MAX_URL_LENGTH,
        name='char_embedding'
    )(input_layer)
    
    # Multiple parallel Conv1D branches with different kernel sizes
    # This captures patterns of different lengths (n-grams)
    conv_outputs = []
    
    for i, (filters, kernel_size) in enumerate(zip(config.CONV_FILTERS, config.KERNEL_SIZES)):
        conv = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            activation='relu',
            padding='same',
            name=f'conv1d_k{kernel_size}'
        )(embedding)
        
        # Batch normalization for stable training
        conv = layers.BatchNormalization(name=f'bn_k{kernel_size}')(conv)
        
        # Global max pooling to get fixed-size output
        pooled = layers.GlobalMaxPooling1D(name=f'maxpool_k{kernel_size}')(conv)
        conv_outputs.append(pooled)
    
    # Concatenate all conv outputs
    if len(conv_outputs) > 1:
        merged = layers.Concatenate(name='concat')(conv_outputs)
    else:
        merged = conv_outputs[0]
    
    # Dense layers
    x = merged
    for i, units in enumerate(config.DENSE_UNITS):
        x = layers.Dense(units, activation='relu', name=f'dense_{i+1}')(x)
        x = layers.BatchNormalization(name=f'bn_dense_{i+1}')(x)
        x = layers.Dropout(config.DROPOUT_RATE, name=f'dropout_{i+1}')(x)
    
    # Output layer
    output = layers.Dense(1, activation='sigmoid', name='output')(x)
    
    # Create model
    model = models.Model(inputs=input_layer, outputs=output, name='PhishingDetector_1DCNN')
    
    return model

def compile_model(model, config):
    """Compile the model with optimizer, loss, and metrics."""
    
    print("\n--- Compiling Model ---")
    
    # Adam optimizer with custom learning rate
    optimizer = keras.optimizers.Adam(learning_rate=config.LEARNING_RATE)
    
    # Compile
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )
    
    print("✓ Model compiled successfully!")
    print(f"  Optimizer: Adam (lr={config.LEARNING_RATE})")
    print(f"  Loss: Binary Crossentropy")
    print(f"  Metrics: Accuracy, Precision, Recall, AUC")
    
    return model

# Build and compile model
with tf.device(DEVICE):
    model = build_1d_cnn_model(config)
    model = compile_model(model, config)

# Model summary
print("\n--- Model Summary ---")
model.summary()

# Calculate model parameters
total_params = model.count_params()
print(f"\nTotal parameters: {total_params:,}")
print(f"Estimated model size: {total_params * 4 / (1024**2):.2f} MB (float32)")

# Visualize model architecture
try:
    keras.utils.plot_model(
        model,
        to_file=os.path.join(config.RESULTS_PATH, 'model_architecture.png'),
        show_shapes=True,
        show_layer_names=True,
        dpi=150
    )
    print("\n✓ Model architecture diagram saved!")
except Exception as e:
    print(f"\n⚠ Could not save model diagram: {e}")


# ==============================================================================
# SECTION 8: TRAINING CALLBACKS
# ==============================================================================

print("\n" + "=" * 80)
print("SECTION 8: TRAINING CALLBACKS")
print("=" * 80)

def create_callbacks(config):
    """Create training callbacks for monitoring and optimization."""
    
    print("\n--- Setting Up Training Callbacks ---")
    
    callback_list = []
    
    # 1. Early Stopping - prevents overfitting
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=config.EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1,
        mode='min'
    )
    callback_list.append(early_stopping)
    print(f"✓ Early Stopping: patience={config.EARLY_STOPPING_PATIENCE}")
    
    # 2. Learning Rate Reduction - reduces LR when validation loss plateaus
    lr_reducer = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=config.LR_REDUCE_FACTOR,
        patience=config.LR_REDUCE_PATIENCE,
        min_lr=1e-7,
        verbose=1,
        mode='min'
    )
    callback_list.append(lr_reducer)
    print(f"✓ LR Reducer: factor={config.LR_REDUCE_FACTOR}, patience={config.LR_REDUCE_PATIENCE}")
    
    # 3. Model Checkpoint - saves best model
    checkpoint = callbacks.ModelCheckpoint(
        filepath=config.MODEL_SAVE_PATH,
        monitor='val_auc',
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
        mode='max'
    )
    callback_list.append(checkpoint)
    print(f"✓ Model Checkpoint: monitoring val_auc")
    
    # 4. TensorBoard logging (optional)
    tensorboard_dir = os.path.join(config.RESULTS_PATH, 'tensorboard_logs')
    os.makedirs(tensorboard_dir, exist_ok=True)
    tensorboard = callbacks.TensorBoard(
        log_dir=tensorboard_dir,
        histogram_freq=1,
        write_graph=True
    )
    callback_list.append(tensorboard)
    print(f"✓ TensorBoard: logs at {tensorboard_dir}")
    
    # 5. Custom callback for logging
    class TrainingLogger(callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            print(f"\n📊 Epoch {epoch+1} Summary:")
            print(f"   Train - Loss: {logs['loss']:.4f}, Acc: {logs['accuracy']:.4f}, AUC: {logs['auc']:.4f}")
            print(f"   Val   - Loss: {logs['val_loss']:.4f}, Acc: {logs['val_accuracy']:.4f}, AUC: {logs['val_auc']:.4f}")
    
    callback_list.append(TrainingLogger())
    print("✓ Custom Training Logger")
    
    return callback_list

# Create callbacks
training_callbacks = create_callbacks(config)

# ==============================================================================
# SECTION 9: MODEL TRAINING
# ==============================================================================

print("\n" + "=" * 80)
print("SECTION 9: MODEL TRAINING")
print("=" * 80)

def train_model(model, X_train, y_train, X_val, y_val, config, callbacks_list):
    """Train the model with the prepared data."""
    
    print("\n--- Starting Model Training ---")
    print(f"Training samples: {len(X_train):,}")
    print(f"Validation samples: {len(X_val):,}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Max epochs: {config.EPOCHS}")
    print(f"Device: {DEVICE}")
    
    # Calculate steps
    steps_per_epoch = len(X_train) // config.BATCH_SIZE
    validation_steps = len(X_val) // config.BATCH_SIZE
    
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Validation steps: {validation_steps}")
    
    print("\n" + "-" * 50)
    print("Training started at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("-" * 50)
    
    start_time = time.time()
    
    # Train the model
    with tf.device(DEVICE):
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=config.EPOCHS,
            batch_size=config.BATCH_SIZE,
            callbacks=callbacks_list,
            verbose=1
        )
    
    training_time = time.time() - start_time
    
    print("\n" + "-" * 50)
    print("Training completed at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print(f"Total training time: {training_time/60:.2f} minutes")
    print("-" * 50)
    
    # Save training history
    np.save(config.HISTORY_SAVE_PATH, history.history)
    print(f"\n✓ Training history saved to: {config.HISTORY_SAVE_PATH}")
    
    return history, training_time

# Train the model
print("\n⏳ Starting training (this will take 15-30 minutes on GPU)...")
try:
    history, training_time = train_model(
        model, X_train, y_train, X_val, y_val, config, training_callbacks
    )
except tf.errors.ResourceExhaustedError:
    print("\n✗ GPU out of memory! Try reducing BATCH_SIZE or TRAIN_SAMPLE_SIZE")
    raise
except KeyboardInterrupt:
    print("\n⚠ Training interrupted by user")
    raise

# ==============================================================================
# SECTION 10: TRAINING VISUALIZATION
# ==============================================================================

print("\n" + "=" * 80)
print("SECTION 10: TRAINING VISUALIZATION")
print("=" * 80)

def plot_training_history(history):
    """Plot training and validation metrics over epochs."""
    
    print("\n--- Plotting Training History ---")
    
    hist = history.history
    epochs = range(1, len(hist['loss']) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Loss
    axes[0, 0].plot(epochs, hist['loss'], 'b-', label='Training Loss', linewidth=2)
    axes[0, 0].plot(epochs, hist['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0, 0].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Accuracy
    axes[0, 1].plot(epochs, hist['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
    axes[0, 1].plot(epochs, hist['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
    axes[0, 1].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Precision & Recall
    axes[1, 0].plot(epochs, hist['precision'], 'b-', label='Training Precision', linewidth=2)
    axes[1, 0].plot(epochs, hist['val_precision'], 'r-', label='Validation Precision', linewidth=2)
    axes[1, 0].plot(epochs, hist['recall'], 'b--', label='Training Recall', linewidth=2)
    axes[1, 0].plot(epochs, hist['val_recall'], 'r--', label='Validation Recall', linewidth=2)
    axes[1, 0].set_title('Precision & Recall', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: AUC
    axes[1, 1].plot(epochs, hist['auc'], 'b-', label='Training AUC', linewidth=2)
    axes[1, 1].plot(epochs, hist['val_auc'], 'r-', label='Validation AUC', linewidth=2)
    axes[1, 1].set_title('Model AUC', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('AUC')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Training History', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_PATH, 'training_history.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✓ Training history plots saved!")
    
    # Print best metrics
    print("\n--- Best Metrics Achieved ---")
    print(f"Best Validation Loss: {min(hist['val_loss']):.4f} (Epoch {np.argmin(hist['val_loss'])+1})")
    print(f"Best Validation Accuracy: {max(hist['val_accuracy']):.4f} (Epoch {np.argmax(hist['val_accuracy'])+1})")
    print(f"Best Validation AUC: {max(hist['val_auc']):.4f} (Epoch {np.argmax(hist['val_auc'])+1})")

# Plot training history
plot_training_history(history)


# ==============================================================================
# SECTION 11: MODEL EVALUATION
# ==============================================================================

print("\n" + "=" * 80)
print("SECTION 11: MODEL EVALUATION")
print("=" * 80)

def evaluate_model(model, X_test, y_test, threshold=0.5):
    """Comprehensive model evaluation on test set."""
    
    print("\n--- Evaluating Model on Test Set ---")
    print(f"Test samples: {len(X_test):,}")
    
    # Get predictions
    print("Generating predictions...")
    y_pred_proba = model.predict(X_test, batch_size=config.BATCH_SIZE, verbose=1)
    y_pred = (y_pred_proba >= threshold).astype(int).flatten()
    
    # Calculate metrics
    print("\n--- Classification Metrics ---")
    
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
    
    # Detailed classification report
    print("\n--- Detailed Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=['Benign', 'Phishing']))
    
    # Store results
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
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Benign', 'Phishing'],
        yticklabels=['Benign', 'Phishing'],
        annot_kws={'size': 14}
    )
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    
    # Add percentages
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            percentage = cm[i, j] / total * 100
            plt.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)',
                    ha='center', va='center', fontsize=10, color='gray')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_PATH, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print confusion matrix details
    tn, fp, fn, tp = cm.ravel()
    print(f"\nConfusion Matrix Details:")
    print(f"  True Negatives (Benign correctly identified): {tn:,}")
    print(f"  False Positives (Benign misclassified as Phishing): {fp:,}")
    print(f"  False Negatives (Phishing misclassified as Benign): {fn:,}")
    print(f"  True Positives (Phishing correctly identified): {tp:,}")
    
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
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.RESULTS_PATH, 'roc_curve.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    # Find optimal threshold
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print(f"\nOptimal threshold: {optimal_threshold:.4f}")
    print(f"At this threshold - TPR: {tpr[optimal_idx]:.4f}, FPR: {fpr[optimal_idx]:.4f}")
    
    print("\n✓ ROC curve saved!")
    
    return optimal_threshold

def plot_precision_recall_curve(y_test, y_pred_proba):
    """Plot Precision-Recall curve."""
    
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    print("\n--- Plotting Precision-Recall Curve ---")
    
    precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, y_pred_proba)
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
# SECTION 12: MODEL SAVING AND LOADING
# ==============================================================================

print("\n" + "=" * 80)
print("SECTION 12: MODEL SAVING AND LOADING")
print("=" * 80)

def save_model(model, config):
    """Save the trained model."""
    
    print("\n--- Saving Model ---")
    
    # Save in Keras format
    model.save(config.MODEL_SAVE_PATH)
    print(f"✓ Model saved to: {config.MODEL_SAVE_PATH}")
    
    # Get file size
    model_size = os.path.getsize(config.MODEL_SAVE_PATH) / (1024**2)
    print(f"  Model size: {model_size:.2f} MB")
    
    # Also save weights separately (optional backup)
    weights_path = config.MODEL_SAVE_PATH.replace('.keras', '.weights.h5')
    model.save_weights(weights_path)
    print(f"✓ Weights saved to: {weights_path}")
    
    return config.MODEL_SAVE_PATH

def load_and_verify_model(model_path, X_test, y_test):
    """Load saved model and verify it works correctly."""
    
    print("\n--- Loading and Verifying Model ---")
    
    try:
        # Load model
        loaded_model = keras.models.load_model(model_path)
        print(f"✓ Model loaded from: {model_path}")
        
        # Verify by making predictions
        print("Verifying model predictions...")
        sample_size = min(1000, len(X_test))
        X_sample = X_test[:sample_size]
        y_sample = y_test[:sample_size]
        
        predictions = loaded_model.predict(X_sample, verbose=0)
        y_pred = (predictions >= 0.5).astype(int).flatten()
        
        accuracy = accuracy_score(y_sample, y_pred)
        print(f"✓ Verification accuracy on {sample_size} samples: {accuracy:.4f}")
        
        print("\n✓ Model loading and verification successful!")
        return loaded_model
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None

# Save the model
saved_model_path = save_model(model, config)

# Load and verify
loaded_model = load_and_verify_model(saved_model_path, X_test, y_test)


# ==============================================================================
# SECTION 13: MODEL EXPLAINABILITY (SHAP)
# ==============================================================================

print("\n" + "=" * 80)
print("SECTION 13: MODEL EXPLAINABILITY (SHAP)")
print("=" * 80)

def explain_model_with_shap(model, X_train, X_test, test_urls, config, num_samples=100):
    """
    Use SHAP to explain model predictions.
    For character-level models, we analyze which character positions are most important.
    """
    
    print("\n--- SHAP Explainability Analysis ---")
    print(f"Using {num_samples} samples for SHAP analysis (memory efficient)")
    
    try:
        # Sample data for SHAP (full dataset is too large)
        np.random.seed(SEED)
        sample_indices = np.random.choice(len(X_test), min(num_samples, len(X_test)), replace=False)
        X_sample = X_test[sample_indices]
        
        # Background data for SHAP
        background_indices = np.random.choice(len(X_train), min(100, len(X_train)), replace=False)
        X_background = X_train[background_indices]
        
        print("Creating SHAP explainer (this may take a few minutes)...")
        
        # Use DeepExplainer for neural networks
        explainer = shap.DeepExplainer(model, X_background)
        
        print("Calculating SHAP values...")
        shap_values = explainer.shap_values(X_sample)
        
        # Handle output format
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        print(f"SHAP values shape: {shap_values.shape}")
        
        # Analyze character position importance
        print("\n--- Character Position Importance ---")
        
        # Average absolute SHAP values across samples for each position
        position_importance = np.mean(np.abs(shap_values), axis=(0, 2)) if len(shap_values.shape) > 2 else np.mean(np.abs(shap_values), axis=0)
        
        # Plot position importance
        plt.figure(figsize=(14, 5))
        plt.bar(range(len(position_importance)), position_importance, color='#3498db', alpha=0.7)
        plt.xlabel('Character Position in URL', fontsize=12)
        plt.ylabel('Mean |SHAP Value|', fontsize=12)
        plt.title('Character Position Importance for Phishing Detection', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Highlight most important positions
        top_positions = np.argsort(position_importance)[-10:][::-1]
        for pos in top_positions:
            plt.bar(pos, position_importance[pos], color='#e74c3c', alpha=0.9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(config.RESULTS_PATH, 'shap_position_importance.png'), dpi=150, bbox_inches='tight')
        plt.show()
        
        print("\nTop 10 Most Important Character Positions:")
        for i, pos in enumerate(top_positions):
            print(f"  {i+1}. Position {pos}: importance = {position_importance[pos]:.6f}")
        
        print("\n✓ SHAP analysis complete!")
        
        return shap_values, position_importance
        
    except Exception as e:
        print(f"\n⚠ SHAP analysis encountered an error: {e}")
        print("This is common with complex models. Falling back to alternative analysis...")
        return None, None

def analyze_url_patterns(test_urls, y_test, y_pred, num_examples=10):
    """Analyze URL patterns for correct and incorrect predictions."""
    
    print("\n--- URL Pattern Analysis ---")
    
    # Find indices for different prediction outcomes
    correct_benign = [i for i in range(len(y_test)) if y_test[i] == 0 and y_pred[i] == 0]
    correct_phishing = [i for i in range(len(y_test)) if y_test[i] == 1 and y_pred[i] == 1]
    false_positives = [i for i in range(len(y_test)) if y_test[i] == 0 and y_pred[i] == 1]
    false_negatives = [i for i in range(len(y_test)) if y_test[i] == 1 and y_pred[i] == 0]
    
    print(f"\nCorrectly classified benign URLs: {len(correct_benign):,}")
    print(f"Correctly classified phishing URLs: {len(correct_phishing):,}")
    print(f"False positives (benign → phishing): {len(false_positives):,}")
    print(f"False negatives (phishing → benign): {len(false_negatives):,}")
    
    # Show examples
    def show_examples(indices, title, num=5):
        print(f"\n{title}:")
        for i, idx in enumerate(indices[:num]):
            if idx < len(test_urls):
                print(f"  {i+1}. {test_urls[idx][:80]}...")
    
    if false_positives:
        show_examples(false_positives, "False Positive Examples (Benign misclassified as Phishing)")
    
    if false_negatives:
        show_examples(false_negatives, "False Negative Examples (Phishing misclassified as Benign)")
    
    # Analyze URL characteristics
    print("\n--- URL Characteristics Analysis ---")
    
    def analyze_urls(urls, label):
        if not urls:
            return
        lengths = [len(url) for url in urls]
        dot_counts = [url.count('.') for url in urls]
        slash_counts = [url.count('/') for url in urls]
        digit_counts = [sum(c.isdigit() for c in url) for url in urls]
        
        print(f"\n{label}:")
        print(f"  Avg length: {np.mean(lengths):.1f}")
        print(f"  Avg dots: {np.mean(dot_counts):.1f}")
        print(f"  Avg slashes: {np.mean(slash_counts):.1f}")
        print(f"  Avg digits: {np.mean(digit_counts):.1f}")
    
    # Sample URLs for analysis
    benign_urls = [test_urls[i] for i in correct_benign[:1000] if i < len(test_urls)]
    phishing_urls = [test_urls[i] for i in correct_phishing[:1000] if i < len(test_urls)]
    
    analyze_urls(benign_urls, "Benign URLs")
    analyze_urls(phishing_urls, "Phishing URLs")

# Run SHAP analysis (with error handling for Colab memory limits)
try:
    shap_values, position_importance = explain_model_with_shap(
        model, X_train, X_test, data_loader.test_urls, config, num_samples=100
    )
except Exception as e:
    print(f"⚠ SHAP analysis skipped due to: {e}")
    shap_values, position_importance = None, None

# Analyze URL patterns
analyze_url_patterns(data_loader.test_urls, y_test, results['y_pred'])

# ==============================================================================
# SECTION 14: PREDICTION FUNCTION FOR NEW URLs
# ==============================================================================

print("\n" + "=" * 80)
print("SECTION 14: PREDICTION FUNCTION FOR NEW URLs")
print("=" * 80)

class PhishingURLDetector:
    """Class for making predictions on new URLs."""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.char_to_idx = {chr(i): i for i in range(config.VOCAB_SIZE)}
    
    def preprocess_url(self, url):
        """Convert a single URL to model input format."""
        sequence = []
        for char in url[:self.config.MAX_URL_LENGTH]:
            idx = ord(char) if ord(char) < self.config.VOCAB_SIZE else 0
            sequence.append(idx)
        
        # Pad sequence
        padded = pad_sequences(
            [sequence],
            maxlen=self.config.MAX_URL_LENGTH,
            padding='post',
            truncating='post',
            value=0
        )
        return padded
    
    def predict(self, url):
        """Predict if a URL is phishing or benign."""
        X = self.preprocess_url(url)
        probability = self.model.predict(X, verbose=0)[0][0]
        
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
        results = []
        for url in urls:
            results.append(self.predict(url))
        return results
    
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
detector = PhishingURLDetector(model, config)

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
# SECTION 15: FINAL SUMMARY AND RESULTS EXPORT
# ==============================================================================

print("\n" + "=" * 80)
print("SECTION 15: FINAL SUMMARY AND RESULTS EXPORT")
print("=" * 80)

def generate_final_report(results, training_time, config):
    """Generate a comprehensive final report."""
    
    print("\n" + "=" * 80)
    print("                    FINAL TRAINING REPORT")
    print("=" * 80)
    
    print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         PHISHING URL DETECTION MODEL                         ║
║                              1D-CNN Architecture                             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  DATASET INFORMATION                                                         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Training Samples:     {config.TRAIN_SAMPLE_SIZE:>10,}                                          ║
║  Validation Split:     {config.VALIDATION_SPLIT*100:>10.0f}%                                          ║
║  Test Samples:         {config.TEST_SAMPLE_SIZE:>10,}                                          ║
║  Max URL Length:       {config.MAX_URL_LENGTH:>10}                                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  MODEL ARCHITECTURE                                                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Embedding Dimension:  {config.EMBEDDING_DIM:>10}                                          ║
║  Conv Filters:         {str(config.CONV_FILTERS):>10}                                   ║
║  Kernel Sizes:         {str(config.KERNEL_SIZES):>10}                                      ║
║  Dense Units:          {str(config.DENSE_UNITS):>10}                                       ║
║  Dropout Rate:         {config.DROPOUT_RATE:>10}                                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  TRAINING CONFIGURATION                                                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Batch Size:           {config.BATCH_SIZE:>10}                                          ║
║  Max Epochs:           {config.EPOCHS:>10}                                          ║
║  Learning Rate:        {config.LEARNING_RATE:>10}                                        ║
║  Training Time:        {training_time/60:>10.2f} min                                     ║
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
    
    # Save results to file
    results_summary = {
        'dataset': {
            'train_samples': config.TRAIN_SAMPLE_SIZE,
            'test_samples': config.TEST_SAMPLE_SIZE,
            'max_url_length': config.MAX_URL_LENGTH
        },
        'model': {
            'embedding_dim': config.EMBEDDING_DIM,
            'conv_filters': config.CONV_FILTERS,
            'kernel_sizes': config.KERNEL_SIZES,
            'dense_units': config.DENSE_UNITS,
            'dropout_rate': config.DROPOUT_RATE
        },
        'training': {
            'batch_size': config.BATCH_SIZE,
            'epochs': config.EPOCHS,
            'learning_rate': config.LEARNING_RATE,
            'training_time_minutes': training_time / 60
        },
        'metrics': {
            'accuracy': results['accuracy'],
            'precision': results['precision'],
            'recall': results['recall'],
            'f1_score': results['f1_score'],
            'roc_auc': results['roc_auc']
        }
    }
    
    # Save as JSON
    import json
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
        (config.HISTORY_SAVE_PATH, "Training History"),
        (os.path.join(config.RESULTS_PATH, 'results_summary.json'), "Results Summary"),
        (os.path.join(config.RESULTS_PATH, 'training_history.png'), "Training History Plot"),
        (os.path.join(config.RESULTS_PATH, 'confusion_matrix.png'), "Confusion Matrix"),
        (os.path.join(config.RESULTS_PATH, 'roc_curve.png'), "ROC Curve"),
        (os.path.join(config.RESULTS_PATH, 'precision_recall_curve.png'), "PR Curve"),
        (os.path.join(config.RESULTS_PATH, 'class_distribution.png'), "Class Distribution"),
        (os.path.join(config.RESULTS_PATH, 'url_length_distribution.png'), "URL Length Distribution"),
        (os.path.join(config.RESULTS_PATH, 'shap_position_importance.png'), "SHAP Analysis"),
    ]
    
    for file_path, description in files_to_check:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path) / 1024  # KB
            print(f"  ✓ {description}: {file_path} ({size:.1f} KB)")
        else:
            print(f"  ✗ {description}: Not found")

# Generate final report
final_results = generate_final_report(results, training_time, config)

# List all saved files
list_saved_files(config)

# ==============================================================================
# SECTION 16: DOWNLOAD FILES (FOR COLAB)
# ==============================================================================

print("\n" + "=" * 80)
print("SECTION 16: DOWNLOAD FILES")
print("=" * 80)

def create_download_zip(config):
    """Create a ZIP file with all results for easy download."""
    
    print("\n--- Creating Download Package ---")
    
    zip_filename = "/content/phishing_detection_results.zip"
    
    try:
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add model
            if os.path.exists(config.MODEL_SAVE_PATH):
                zipf.write(config.MODEL_SAVE_PATH, os.path.basename(config.MODEL_SAVE_PATH))
            
            # Add all results
            if os.path.exists(config.RESULTS_PATH):
                for root, dirs, files in os.walk(config.RESULTS_PATH):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.join('results', file)
                        zipf.write(file_path, arcname)
            
            # Add training history
            if os.path.exists(config.HISTORY_SAVE_PATH):
                zipf.write(config.HISTORY_SAVE_PATH, os.path.basename(config.HISTORY_SAVE_PATH))
        
        zip_size = os.path.getsize(zip_filename) / (1024**2)
        print(f"✓ Download package created: {zip_filename} ({zip_size:.2f} MB)")
        
        # Provide download link in Colab
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
🎉 Congratulations! Your phishing URL detection model has been trained successfully.

📊 Model Performance:
   • Accuracy:  {results['accuracy']:.2%}
   • Precision: {results['precision']:.2%}
   • Recall:    {results['recall']:.2%}
   • F1-Score:  {results['f1_score']:.2%}
   • ROC-AUC:   {results['roc_auc']:.4f}

📁 Files Saved:
   • Model: {config.MODEL_SAVE_PATH}
   • Results: {config.RESULTS_PATH}/
   • Download: /content/phishing_detection_results.zip

🔮 To use the model for predictions:
   detector = PhishingURLDetector(model, config)
   result = detector.predict("suspicious-url.com")

📝 For your FYP report, all visualizations are saved in the results folder.

Good luck with your Final Year Project! 🚀
""")
print("=" * 80)
