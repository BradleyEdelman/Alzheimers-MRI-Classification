# Alzheimer's Disease Classification from Anatomical MRI Images
This repository contains a series of Jupyter notebooks focused on classifying Alzheimer's disease (AD) using anatomical MRI images. The project explores various machine learning models and strategies for data exploration, preprocessing, and model evaluation. All notebooks were executed on Databricks with AWS EC2 instances. <br />

## Dataset
https://www.kaggle.com/datasets/borhanitrash/alzheimer-mri-disease-classification-dataset <br />

# Notebooks
## 00 Data Cleaning
This notebook covers data cleaning techniques, addressing class imbalance, and applying Principal Component Analysis (PCA) for dimensionality reduction. <br />

## 01 Random Forest
This notebook explores parallel processing using dask to improve computation efficiency with a Random Forest classifier. The model's performance is estimated using chance and permutation testing, and feature importance is evaluated. <br />

## 02 CNN
This notebook implements a custom Convolutional Neural Network (CNN) architecture, along with data augmentation techniques and hyperparameter tuning. <br />

## 03 ResNet50 Transfer Learning
This notebook utilizes the ResNet50 model, which is fine-tuned (transfer learning) to improve classification accuracy. <br />

# Requirements
To reproduce this project, see the dependencies in the requirements.txt file

# License
This project is licensed under the MIT License - see the LICENSE file for details.
