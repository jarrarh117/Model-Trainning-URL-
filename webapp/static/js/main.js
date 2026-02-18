/**
 * ============================================================================
 * PHISHGUARD - PHISHING URL DETECTION WEB APPLICATION
 * Frontend JavaScript - XGBoost Model
 * ============================================================================
 */

// ============================================================================
// CONFIGURATION
// ============================================================================
const API_BASE = '';  // Same origin

// ============================================================================
// STATE
// ============================================================================
let state = {
    isLoading: false,
    modelAvailable: true  // Default to true, update from status check
};

// ============================================================================
// INITIALIZATION
// ============================================================================
document.addEventListener('DOMContentLoaded', function() {
    console.log('PhishGuard: Initializing...');
    init();
});

function init() {
    console.log('PhishGuard: Setting up...');
    checkModelStatus();
    setupEventListeners();
}

// ============================================================================
// API FUNCTIONS
// ============================================================================

async function checkModelStatus() {
    console.log('PhishGuard: Checking model status...');
    try {
        const response = await fetch(`${API_BASE}/api/status`);
        const data = await response.json();
        console.log('PhishGuard: Status response:', data);
        
        state.modelAvailable = data.xgboost_available;
        updateStatusBadge();
        
        if (state.modelAvailable) {
            showToast('XGBoost model ready!', 'success');
        } else {
            showToast('Model not available', 'error');
        }
    } catch (error) {
        console.error('PhishGuard: Failed to check model status:', error);
        showToast('Failed to connect to server', 'error');
        // Keep modelAvailable as true to allow attempts
        state.modelAvailable = true;
    }
}

async function predictUrl(url) {
    console.log('PhishGuard: Predicting URL:', url);
    const response = await fetch(`${API_BASE}/api/predict`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ url: url })
    });
    
    console.log('PhishGuard: Response status:', response.status);
    
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Prediction failed');
    }
    
    return response.json();
}

async function predictBatch(urls) {
    const response = await fetch(`${API_BASE}/api/predict/batch`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ urls: urls })
    });
    
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || 'Batch prediction failed');
    }
    
    return response.json();
}

// ============================================================================
// EVENT LISTENERS
// ============================================================================

function setupEventListeners() {
    console.log('PhishGuard: Setting up event listeners...');
    
    // URL input - Enter key
    const urlInput = document.getElementById('url-input');
    if (urlInput) {
        urlInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                console.log('PhishGuard: Enter key pressed');
                handleScan();
            }
        });
    }
    
    // Clear button
    const clearBtn = document.getElementById('clear-btn');
    if (clearBtn) {
        clearBtn.addEventListener('click', function() {
            if (urlInput) {
                urlInput.value = '';
                urlInput.focus();
            }
        });
    }
    
    // Scan button
    const scanBtn = document.getElementById('scan-btn');
    if (scanBtn) {
        console.log('PhishGuard: Scan button found, adding click listener');
        scanBtn.addEventListener('click', function() {
            console.log('PhishGuard: Scan button clicked');
            handleScan();
        });
    } else {
        console.error('PhishGuard: Scan button NOT found!');
    }
    
    // Example buttons
    const exampleBtns = document.querySelectorAll('.example-btn');
    exampleBtns.forEach(function(btn) {
        btn.addEventListener('click', function() {
            if (urlInput) {
                urlInput.value = btn.dataset.url;
                urlInput.focus();
            }
        });
    });
    
    // Batch toggle
    const batchToggle = document.getElementById('batch-toggle');
    const batchSection = document.querySelector('.batch-section');
    if (batchToggle && batchSection) {
        batchToggle.addEventListener('click', function() {
            batchSection.classList.toggle('expanded');
        });
    }
    
    // Batch scan
    const batchScanBtn = document.getElementById('batch-scan-btn');
    if (batchScanBtn) {
        batchScanBtn.addEventListener('click', handleBatchScan);
    }
    
    console.log('PhishGuard: Event listeners setup complete');
}

// ============================================================================
// UI FUNCTIONS
// ============================================================================

function updateStatusBadge() {
    const xgbStatus = document.getElementById('xgb-status');
    if (!xgbStatus) return;
    
    const xgbDot = xgbStatus.querySelector('.status-dot');
    if (!xgbDot) return;
    
    xgbDot.classList.remove('loading');
    
    if (state.modelAvailable) {
        xgbDot.classList.add('online');
        xgbDot.classList.remove('offline');
    } else {
        xgbDot.classList.add('offline');
        xgbDot.classList.remove('online');
    }
}

function setLoading(loading) {
    state.isLoading = loading;
    const scanBtn = document.getElementById('scan-btn');
    
    if (scanBtn) {
        if (loading) {
            scanBtn.classList.add('loading');
            scanBtn.disabled = true;
        } else {
            scanBtn.classList.remove('loading');
            scanBtn.disabled = false;
        }
    }
}

function showToast(message, type) {
    type = type || 'info';
    console.log('PhishGuard Toast:', type, message);
    
    const toast = document.getElementById('toast');
    if (!toast) return;
    
    const icon = toast.querySelector('.toast-icon');
    const msg = toast.querySelector('.toast-message');
    
    if (!icon || !msg) return;
    
    // Set icon based on type
    const icons = {
        success: 'fas fa-check-circle',
        error: 'fas fa-exclamation-circle',
        warning: 'fas fa-exclamation-triangle',
        info: 'fas fa-info-circle'
    };
    
    icon.className = 'toast-icon ' + (icons[type] || icons.info);
    msg.textContent = message;
    
    toast.className = 'toast show ' + type;
    
    setTimeout(function() {
        toast.classList.remove('show');
    }, 3000);
}

// ============================================================================
// SCAN HANDLERS
// ============================================================================

async function handleScan() {
    console.log('PhishGuard: handleScan called');
    
    const urlInput = document.getElementById('url-input');
    if (!urlInput) {
        console.error('PhishGuard: URL input not found');
        return;
    }
    
    const url = urlInput.value.trim();
    console.log('PhishGuard: URL to scan:', url);
    
    if (!url) {
        showToast('Please enter a URL', 'warning');
        urlInput.focus();
        return;
    }
    
    setLoading(true);
    
    try {
        console.log('PhishGuard: Making prediction request...');
        const result = await predictUrl(url);
        console.log('PhishGuard: Prediction result:', result);
        displayResult(result);
        showToast('Scan complete!', 'success');
    } catch (error) {
        console.error('PhishGuard: Prediction error:', error);
        showToast(error.message || 'Scan failed', 'error');
    } finally {
        setLoading(false);
    }
}

async function handleBatchScan() {
    const batchInput = document.getElementById('batch-input');
    if (!batchInput) return;
    
    const text = batchInput.value.trim();
    
    if (!text) {
        showToast('Please enter URLs', 'warning');
        batchInput.focus();
        return;
    }
    
    const urls = text.split('\n').map(function(u) { return u.trim(); }).filter(function(u) { return u; });
    
    if (urls.length === 0) {
        showToast('No valid URLs found', 'warning');
        return;
    }
    
    const batchScanBtn = document.getElementById('batch-scan-btn');
    if (batchScanBtn) {
        batchScanBtn.disabled = true;
        batchScanBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Scanning...';
    }
    
    try {
        const data = await predictBatch(urls);
        displayBatchResults(data.results);
    } catch (error) {
        showToast(error.message, 'error');
    } finally {
        if (batchScanBtn) {
            batchScanBtn.disabled = false;
            batchScanBtn.innerHTML = '<i class="fas fa-layer-group"></i> Scan All URLs';
        }
    }
}

// ============================================================================
// RESULT DISPLAY
// ============================================================================

function displayResult(result) {
    console.log('PhishGuard: Displaying result...');
    
    const resultsSection = document.getElementById('results-section');
    const resultCard = document.getElementById('result-card');
    
    if (!resultsSection || !resultCard) {
        console.error('PhishGuard: Result elements not found');
        return;
    }
    
    resultsSection.classList.add('show');
    resultsSection.style.display = 'block';
    
    const isPhishing = result.is_phishing;
    const probability = (result.probability * 100).toFixed(1);
    const confidence = (result.confidence * 100).toFixed(1);
    const safeProb = (100 - parseFloat(probability)).toFixed(1);
    
    // Get risk level
    let riskLevel, riskClass, riskIcon, riskDesc, verdictIcon, verdictText;
    if (result.probability >= 0.8) {
        riskLevel = 'HIGH RISK';
        riskClass = 'high';
        riskIcon = 'fas fa-skull-crossbones';
        riskDesc = 'This URL shows strong characteristics of a phishing attempt.';
        verdictIcon = 'fas fa-times-circle';
        verdictText = '⚠️ <strong>Do NOT visit this URL!</strong> This site has a very high probability of being a phishing attack designed to steal your personal information.';
    } else if (result.probability >= 0.5) {
        riskLevel = 'MEDIUM RISK';
        riskClass = 'medium';
        riskIcon = 'fas fa-exclamation-triangle';
        riskDesc = 'This URL has suspicious characteristics that warrant caution.';
        verdictIcon = 'fas fa-exclamation-circle';
        verdictText = '⚠️ <strong>Proceed with caution!</strong> This URL shows some phishing indicators. Verify the source before entering any personal information.';
    } else if (result.probability >= 0.2) {
        riskLevel = 'LOW RISK';
        riskClass = 'low';
        riskIcon = 'fas fa-shield-alt';
        riskDesc = 'This URL has minimal suspicious characteristics.';
        verdictIcon = 'fas fa-info-circle';
        verdictText = 'ℹ️ This URL appears mostly safe but has some minor suspicious traits. Still, verify you trust the source.';
    } else {
        riskLevel = 'SAFE';
        riskClass = 'safe';
        riskIcon = 'fas fa-check-circle';
        riskDesc = 'This URL shows characteristics of a legitimate website.';
        verdictIcon = 'fas fa-check-circle';
        verdictText = '✅ <strong>This URL appears safe!</strong> Our analysis indicates this is likely a legitimate website. As always, stay vigilant online.';
    }
    
    // Build risk indicators based on URL features
    let riskIndicatorsHTML = buildRiskIndicators(result.url_features, result.probability);
    
    // Build top features HTML
    let featuresHTML = '';
    if (result.top_features && result.top_features.length > 0) {
        let featureItems = result.top_features.map(function(f) {
            let importance = Math.min(f.importance * 100, 100);
            return '<div class="feature-item">' +
                '<span class="feature-name">' + formatFeatureName(f.name) + '</span>' +
                '<div class="feature-bar"><div class="feature-fill" style="width: ' + importance + '%"></div></div>' +
            '</div>';
        }).join('');
        
        featuresHTML = '<div class="features-section">' +
            '<h4><i class="fas fa-chart-bar"></i> Top Contributing Features</h4>' +
            '<div class="features-list">' + featureItems + '</div>' +
        '</div>';
    }
    
    // Build URL analysis HTML with more details
    let urlFeaturesHTML = buildUrlAnalysis(result.url_features);
    
    resultCard.innerHTML = 
        '<div class="result-content ' + riskClass + '">' +
            // Header Section
            '<div class="result-header">' +
                '<div class="result-icon ' + riskClass + '">' +
                    '<i class="' + (isPhishing ? 'fas fa-exclamation-triangle' : 'fas fa-shield-alt') + '"></i>' +
                '</div>' +
                '<div class="result-title">' +
                    '<h3>' + result.label + '</h3>' +
                    '<span class="risk-badge ' + riskClass + '">' +
                        '<i class="' + riskIcon + '"></i> ' + riskLevel +
                    '</span>' +
                '</div>' +
            '</div>' +
            
            // URL Display
            '<div class="result-url">' +
                '<i class="fas fa-link"></i>' +
                '<span>' + escapeHtml(result.url) + '</span>' +
            '</div>' +
            
            // Statistics Section
            '<div class="result-stats">' +
                '<div class="stat-circle">' +
                    '<svg viewBox="0 0 36 36" class="circular-chart">' +
                        '<path class="circle-bg" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"/>' +
                        '<path class="circle ' + riskClass + '" stroke-dasharray="' + probability + ', 100" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"/>' +
                    '</svg>' +
                    '<div class="stat-center">' +
                        '<span class="stat-value">' + probability + '%</span>' +
                        '<span class="stat-label">Risk Score</span>' +
                    '</div>' +
                '</div>' +
                '<div class="stat-details">' +
                    '<div class="stat-row">' +
                        '<span class="stat-name"><i class="fas fa-shield-virus"></i> Phishing Probability</span>' +
                        '<span class="stat-val" style="color: ' + (isPhishing ? '#ef4444' : '#10b981') + '">' + probability + '%</span>' +
                    '</div>' +
                    '<div class="stat-row">' +
                        '<span class="stat-name"><i class="fas fa-shield-alt"></i> Safe Probability</span>' +
                        '<span class="stat-val" style="color: ' + (!isPhishing ? '#10b981' : '#ef4444') + '">' + safeProb + '%</span>' +
                    '</div>' +
                    '<div class="stat-row">' +
                        '<span class="stat-name"><i class="fas fa-bullseye"></i> Model Confidence</span>' +
                        '<span class="stat-val">' + confidence + '%</span>' +
                    '</div>' +
                    '<div class="stat-row">' +
                        '<span class="stat-name"><i class="fas fa-robot"></i> Analysis Engine</span>' +
                        '<span class="stat-val">' + result.model + '</span>' +
                    '</div>' +
                    '<div class="stat-row">' +
                        '<span class="stat-name"><i class="fas fa-clock"></i> Scan Time</span>' +
                        '<span class="stat-val">' + new Date().toLocaleTimeString() + '</span>' +
                    '</div>' +
                '</div>' +
            '</div>' +
            
            // Risk Indicators
            riskIndicatorsHTML +
            
            // URL Analysis Features
            urlFeaturesHTML +
            
            // Top Contributing Features
            featuresHTML +
            
            // Verdict Section
            '<div class="verdict-section ' + riskClass + '">' +
                '<div class="verdict-icon"><i class="' + verdictIcon + '"></i></div>' +
                '<p class="verdict-text">' + verdictText + '</p>' +
            '</div>' +
        '</div>';
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function buildRiskIndicators(urlFeatures, probability) {
    if (!urlFeatures) return '';
    
    let indicators = [];
    
    // Check various risk factors
    if (urlFeatures.has_ip === 1 || urlFeatures.ip_in_url === 1) {
        indicators.push({type: 'danger', icon: 'fas fa-network-wired', text: 'IP address in URL'});
    }
    if (urlFeatures.has_at_symbol === 1 || urlFeatures.at_symbol === 1) {
        indicators.push({type: 'danger', icon: 'fas fa-at', text: '@ symbol detected'});
    }
    if (urlFeatures.suspicious_tld === 1) {
        indicators.push({type: 'warning', icon: 'fas fa-globe', text: 'Suspicious TLD'});
    }
    if ((urlFeatures.num_subdomains || 0) > 3) {
        indicators.push({type: 'warning', icon: 'fas fa-sitemap', text: 'Many subdomains (' + urlFeatures.num_subdomains + ')'});
    }
    if ((urlFeatures.url_length || 0) > 100) {
        indicators.push({type: 'warning', icon: 'fas fa-ruler-horizontal', text: 'Long URL (' + urlFeatures.url_length + ' chars)'});
    }
    if (urlFeatures.has_https === 0 || urlFeatures.https === 0) {
        indicators.push({type: 'warning', icon: 'fas fa-unlock', text: 'No HTTPS'});
    } else {
        indicators.push({type: 'safe', icon: 'fas fa-lock', text: 'HTTPS enabled'});
    }
    if ((urlFeatures.num_special_chars || 0) > 5) {
        indicators.push({type: 'warning', icon: 'fas fa-hashtag', text: 'Many special characters'});
    }
    if (urlFeatures.has_port === 1 || urlFeatures.port === 1) {
        indicators.push({type: 'warning', icon: 'fas fa-door-open', text: 'Custom port in URL'});
    }
    if ((urlFeatures.path_length || 0) < 10 && probability < 0.3) {
        indicators.push({type: 'safe', icon: 'fas fa-folder', text: 'Normal path structure'});
    }
    
    if (indicators.length === 0) return '';
    
    let indicatorHTML = indicators.slice(0, 6).map(function(ind) {
        return '<div class="risk-indicator ' + ind.type + '">' +
            '<div class="risk-indicator-icon"><i class="' + ind.icon + '"></i></div>' +
            '<span class="risk-indicator-text">' + ind.text + '</span>' +
        '</div>';
    }).join('');
    
    return '<div class="risk-indicators">' + indicatorHTML + '</div>';
}

function buildUrlAnalysis(urlFeatures) {
    if (!urlFeatures) return '';
    
    // Select key features to display
    let keyFeatures = [
        {key: 'url_length', label: 'URL Length', icon: 'fas fa-ruler'},
        {key: 'hostname_length', label: 'Hostname Length', icon: 'fas fa-server'},
        {key: 'path_length', label: 'Path Length', icon: 'fas fa-folder-open'},
        {key: 'num_subdomains', label: 'Subdomains', icon: 'fas fa-sitemap'},
        {key: 'num_dots', label: 'Dots Count', icon: 'fas fa-circle'},
        {key: 'num_hyphens', label: 'Hyphens Count', icon: 'fas fa-minus'},
        {key: 'num_digits', label: 'Digit Count', icon: 'fas fa-hashtag'},
        {key: 'num_params', label: 'URL Parameters', icon: 'fas fa-question-circle'}
    ];
    
    let featureItems = keyFeatures.map(function(f) {
        let value = urlFeatures[f.key];
        if (value === undefined || value === null) return '';
        let displayVal = typeof value === 'number' ? (Number.isInteger(value) ? value : value.toFixed(2)) : value;
        return '<div class="url-feature">' +
            '<span class="uf-label"><i class="' + f.icon + '" style="margin-right: 0.5rem; color: var(--primary-light);"></i>' + f.label + '</span>' +
            '<span class="uf-value">' + displayVal + '</span>' +
        '</div>';
    }).filter(function(item) { return item !== ''; }).join('');
    
    if (!featureItems) return '';
    
    return '<div class="url-features-section">' +
        '<h4><i class="fas fa-microscope"></i> URL Structure Analysis</h4>' +
        '<div class="url-features-grid">' + featureItems + '</div>' +
    '</div>';
}

function displayBatchResults(results) {
    const batchResultsSection = document.getElementById('batch-results-section');
    const batchSummary = document.getElementById('batch-summary');
    const batchResultsList = document.getElementById('batch-results-list');
    
    if (!batchResultsSection || !batchSummary || !batchResultsList) return;
    
    batchResultsSection.classList.add('show');
    batchResultsSection.style.display = 'block';
    
    const phishingCount = results.filter(function(r) { return r.is_phishing; }).length;
    const safeCount = results.length - phishingCount;
    
    batchSummary.innerHTML = 
        '<div class="summary-stat phishing"><i class="fas fa-exclamation-triangle"></i><span class="count">' + phishingCount + '</span><span class="label">Phishing</span></div>' +
        '<div class="summary-stat safe"><i class="fas fa-shield-alt"></i><span class="count">' + safeCount + '</span><span class="label">Safe</span></div>' +
        '<div class="summary-stat total"><i class="fas fa-list"></i><span class="count">' + results.length + '</span><span class="label">Total</span></div>';
    
    batchResultsList.innerHTML = results.map(function(result, index) {
        if (result.error) {
            return '<div class="batch-result-item error"><span class="batch-index">' + (index + 1) + '</span><span class="batch-url">' + escapeHtml(result.url) + '</span><span class="batch-error">' + result.error + '</span></div>';
        }
        
        const prob = (result.probability * 100).toFixed(1);
        const isPhishing = result.is_phishing;
        
        return '<div class="batch-result-item ' + (isPhishing ? 'phishing' : 'safe') + '"><span class="batch-index">' + (index + 1) + '</span><span class="batch-url">' + escapeHtml(result.url) + '</span><span class="batch-prob">' + prob + '%</span><span class="batch-label ' + (isPhishing ? 'phishing' : 'safe') + '"><i class="' + (isPhishing ? 'fas fa-exclamation-triangle' : 'fas fa-check-circle') + '"></i> ' + result.label + '</span></div>';
    }).join('');
    
    // Scroll to results
    batchResultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

function formatFeatureName(name) {
    return name
        .replace(/_/g, ' ')
        .replace(/([A-Z])/g, ' $1')
        .replace(/^./, function(str) { return str.toUpperCase(); })
        .trim();
}

function escapeHtml(text) {
    var div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
