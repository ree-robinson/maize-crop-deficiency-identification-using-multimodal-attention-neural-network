# Multimodal Attention Neural Networks for Maize Crop Nutrient Deficiency Identification

## Overview

This project focuses on the identification of nutrient deficiencies in maize crops using a multimodal deep learning approach. The system combines maize leaf image analysis and soil sensor data through an attention-based neural network to improve classification accuracy and reliability.

The model integrates:

- Leaf image features extracted using ResNet-50
- Soil sensor parameters including Nitrogen (N), Phosphorus (P), Potassium (K), pH, and Moisture
- Attention-based multimodal fusion for adaptive feature learning

The proposed system achieved a classification accuracy of **96.12%** for maize nutrient deficiency detection.

## Problem Statement

Traditional nutrient deficiency detection methods rely on manual inspection and laboratory testing, which are time-consuming, expensive, and difficult to scale for large agricultural fields.

Image-only systems often fail when different nutrient deficiencies exhibit visually similar symptoms. To address this limitation, this project combines visual crop information with soil sensor data to improve prediction accuracy.

## Objectives

- Develop a multimodal deep learning framework for maize nutrient deficiency identification
- Combine image and sensor modalities effectively
- Improve classification accuracy using an attention mechanism
- Build a scalable solution for precision agriculture applications
- 
## Proposed Methodology

The proposed system uses two parallel input branches:

### 1. Image Processing Branch
- Maize leaf images are processed using a pre-trained ResNet-50 model
- Extracts deep visual features from crop images

### 2. Sensor Processing Branch
- Soil sensor values are processed using a fully connected neural network
- Inputs include:
  - Nitrogen (N)
  - Phosphorus (P)
  - Potassium (K)
  - Soil pH
  - Moisture

### 3. Attention-Based Fusion
- Image and sensor features are concatenated
- An attention mechanism learns the importance of each feature
- Final classification predicts the nutrient deficiency category

## Dataset

The dataset contains maize leaf images and corresponding soil sensor values.
https://drive.google.com/file/d/1TnoZ139LVNR4kE-BJ8JLtlgcwhL40o1W/view?usp=sharing
test data set
https://drive.google.com/file/d/1LboUmpHpW2uXhaqTJhBNHCDN6vtpHC2g/view?usp=sharing

### Classes

- Healthy
- Nitrogen Deficiency
- Phosphorus Deficiency
- Potassium Deficiency
- Multiple Nutrient Deficiencies


## Preprocessing

### Image Preprocessing
- Resize to `224 × 224`
- ImageNet normalization
- Data augmentation:
  - Horizontal flip
  - Rotation
  - Zoom
  - Color jitter

### Sensor Data Preprocessing
- Z-score normalization
- Feature scaling

## Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Pandas
- Matplotlib
- PIL
- ResNet-50


## Model Performance

| Metric | Value |
|--------|--------|
| Accuracy | 96.12% |
| Precision | 0.962 |
| Recall | 0.961 |
| F1 Score | 0.961 |
| RMSE | 0.095 |
| MAE | 0.070 |

## Comparative Results

| Model | Accuracy |
|-------|-----------|
| Sensor-only Model | 81.7% |
| Image-only ResNet-50 | 89.3% |
| Image + Sensor (Without Attention) | 93.5% |
| Proposed Attention-Based Model | **96.12%** |


## Key Contributions

- Introduced a multimodal attention-based framework for maize nutrient deficiency detection
- Improved classification accuracy using image and sensor fusion
- Demonstrated the effectiveness of attention mechanisms in precision agriculture
- Achieved robust performance with minimal overfitting

## Future Enhancements

- Multi-crop nutrient deficiency detection
- Fertilizer recommendation system
- Mobile application deployment
- Edge AI implementation
- Real-time smart farming integration



## License

This project is developed for academic and research purposes.
