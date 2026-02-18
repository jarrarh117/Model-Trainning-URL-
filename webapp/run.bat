@echo off
echo ============================================
echo    PhishGuard - Phishing URL Detection
echo ============================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo Checking dependencies...
echo.

REM Install requirements if needed
pip install -q flask flask-cors tensorflow xgboost scikit-learn pandas numpy tldextract

echo.
echo Starting PhishGuard Web Application...
echo.
echo ============================================
echo    Open your browser and go to:
echo    http://localhost:5000
echo ============================================
echo.
echo Press Ctrl+C to stop the server
echo.

python app.py
