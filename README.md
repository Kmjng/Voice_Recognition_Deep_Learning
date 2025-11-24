# AI Generated Music Detection 🎵🤖

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Deep learning model to distinguish AI-generated music from authentic human vocals

## 📌 Project Overview

This project develops a CNN-based classifier to identify AI-generated cover songs that use artists' voices without authorization. With the rise of AI music generation tools, protecting artists' vocal copyrights has become increasingly important.

**Duration**: July 2024 - August 2024  
**Team**: 6 members  
**Dataset**: Original songs vs AI cover versions (

## 🎯 Motivation

- **Rising AI cover trend**: Massive views on YouTube AI cover videos
- **Copyright concerns**: Unauthorized use of artists' voices for training
- **Legal gaps**: Current laws don't adequately protect vocal characteristics
- **Need for detection**: Automated systems to identify AI-generated content

## 🔧 Tech Stack

- **Language**: Python 3.8+
- **Deep Learning**: TensorFlow/Keras
- **Audio Processing**: Librosa
- **Visualization**: UMAP, Matplotlib
- **Feature Extraction**: MFCC (Mel-Frequency Cepstral Coefficients)

## 🏗️ Architecture

### Feature Extraction Pipeline  
```
Audio Input → Digital Conversion → Framing → Windowing 
→ FFT → Mel Filter Bank → Log Transform → DCT → MFCC Features
```
### Model
- **Type**: 1D Convolutional Neural Network (CNN)
- **Input**: MFCC feature vectors (101-dimensional)
- **Architecture**: 
  - Conv1D layers with ReLU activation
  - MaxPooling1D for dimensionality reduction
  - Dense layers with Dropout (0.5)
  - Binary classification output (AI vs Original)

## 📊 Key Results

- **CNN Model**: Achieved stable performance with ~95% training accuracy
- **vs DNN**: CNN outperformed simple DNN due to convolutional feature extraction
- **Visualization**: UMAP projection clearly separates AI from original audio features
