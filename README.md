# AI Generated Music Detection 🎵🤖

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Deep learning model to distinguish AI-generated music from authentic human vocals

## 📌 Project Overview

This project develops a CNN-based classifier to identify AI-generated cover songs that use artists' voices without authorization. With the rise of AI music generation tools, protecting artists' vocal copyrights has become increasingly important.

**Duration**: July 2024 - August 2024  
**Team**: 6 members  
**Dataset**: Original songs vs AI cover versions  
🎧 [Listen to sample audio here](https://kmjng.netlify.app/second)

## 🎯 Motivation

- **Rising AI cover trend**: Massive views on YouTube AI cover videos
- **Copyright concerns**: Unauthorized use of artists' voices for training
- **Legal gaps**: Current laws don't adequately protect vocal characteristics
- **Need for detection**: Automated systems to identify AI-generated content

## 🔧 Tech Stack

- **Language**: Python 3.10+
- **Deep Learning**: TensorFlow/Keras
- **Audio Processing**: Librosa
- **Feature Extraction**: **MFCC** (Mel-Frequency Cepstral Coefficients) and **GFCC** (Gammatone Frequency Cepstral Coefficients)
- **Visualization**: UMAP, Matplotlib
- **Feature Extraction**: MFCC (Mel-Frequency Cepstral Coefficients)

## 🏗️ Architecture

### Feature Extraction Pipeline  
```
Audio Input → MFCC Extraction (100 coefficients) → Temporal Averaging 
→ Zero Crossing Rate → 101-D Feature Vector
```
### Model
- **Type**: 1D Convolutional Neural Network (CNN)
- **Input**: MFCC feature vectors (101-dimensional)
- **Architecture**: 
  - 3 x Conv1D layers (ReLU activation)
  - 3 x MaxPooling1D for dimensionality reduction
  - Dense layers with Dropout (0.5)
  - Binary classification output (Sigmoid activation for AI vs Original)

## 📊 Key Results

- **CNN Model**: Achieved stable performance with ~**98%** training accuracy.
- **Robustness**: MFCC와 GFCC 모두 사용하여 특징 추출의 다양성을 시도했습니다.
- **Generalization**: Early Stopping을 적용하여 과적합을 방지하고 테스트 데이터셋에서 **80~85%**의 검증 정확도를 보였습니다.
