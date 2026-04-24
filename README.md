# ai-image-anomaly-detection
Gradient &amp; Frequency-Based Explainable Detection of AI-Generated Images
# Gradient & Frequency-Based Explainable Detection of AI-Generated Images

## 🔍 Problem
Can we detect AI-generated images without training a machine learning model?

## 🧠 Approach
- Gradient analysis (Sobel filters)
- Multi-scale feature extraction
- Frequency-domain analysis (FFT)
- Direction consistency (coherence)
- Patch-wise anomaly detection
- Heatmap visualization

## 📊 Results

### Real Image:
- Mean Absolute Anomaly: 0.249
- High Anomaly Ratio: 0.064
- Avg Direction Coherence: 0.579

### AI Image:
- Mean Absolute Anomaly: 0.175
- High Anomaly Ratio: 0.022
- Avg Direction Coherence: 0.565

## 💡 Key Insight
Real images showed higher anomaly due to natural texture complexity, while AI images appeared smoother.

This highlights a limitation:
Low-level features capture texture complexity rather than purely AI artifacts.

## 🛠️ Tech Stack
- Python
- OpenCV
- NumPy
- Matplotlib

## 🚀 Future Work
- Add ML model (SVM/CNN)
- Improve feature design
- Use real AI datasets
