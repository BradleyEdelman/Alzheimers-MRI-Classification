# Alzheimer's Disease Classification from Anatomical MRI Images <br /> <br />

## Project Overview

This repository contains a series of Jupyter notebooks aimed at classifying Alzheimer's Disease (AD) using anatomical MRI images. The project explores various machine learning techniques, including traditional models like Random Forest and deep learning approaches such as CNNs and ResNet50 transfer learning, to predict AD.

### Key Highlights:
- **Data Exploration & Preprocessing**: Checking data quality, dimensionality reduction via PCA, and addressing class imbalances.
- **Modeling & Evaluation**: Training and evaluating multiple ML models, including Random Forest, CNN, ResNet50, with complementary techniques such as permutation testing, hyperparameter tuning, and transfer learning.
- **Model Explainability**: Using PCA maps and SHAP values to interpret model predictions and identify key features of AD progression.
- **Infrastructure**: Executed on Databricks with AWS EC2 for scalable computing and parallel processing.

This project not only focuses on AD classification but also on understanding how MRI features influence the predictions made by machine learning models.

## Dataset
The dataset used for this project contains labeled anatomical MRI images for Alzheimer's disease classification, and is available on Kaggle:
[Alzheimer MRI Disease Classification Dataset](https://www.kaggle.com/datasets/borhanitrash/alzheimer-mri-disease-classification-dataset) <br /> <br />

# Notebooks
## 00 Data Cleaning
Prepares the dataset for classification by:
- Accessing Parquet data from an AWS S3 bucket.
- Formatting and standardizing training data.
- Addressing class imbalance using synthetic data generation techniques.
- Preprocessing test data for evaluation. <br />

## 01 Random Forest
Implements a random forest classifier to classify Alzheimer's disease, with steps including:
- Applying Principal Component Analysis (PCA) for dimensionality reduction and feature extraction
- Cross-validation of classification results
- Permutation testing for statistical significance using parallel processing.
- Exploring feature importance <br />

## 02 Custom CNN
Develops a custom convolutional neural network (CNN) to classify Alzheimer's disease, focusing on:
- Hyperparameter tuning to optimize model performance.
- Distributed training using TensorFlow's MirroredStrategy.
- The effect of class imbalance on class-specific classification accuracy. <br />

## 03 ResNet50 Transfer Learning
Utilizes the ResNet50 model for transfer learning, exploring:
- Fine-tuning the model to improve classification accuracy.
- Prediction accuracy and complexity metrics as a function of model pruning. <br />

## 04 Model Explainability
Explores the explainability of the fine-tuned ResNet50 model using SHapley Additive exPlanations (SHAP):
- Identifying key spatial features of Alzheimer's progression.
- Investigating misclassifications and performing cluster analyses.
- Validating model focus and reliability through SHAP visualizations. <br /> <br />

# Results
<!--
## Model Performance
This section briefly summarizes the performance of the different models and appraoches used in the project:

- **Random Forest**: Achieved an accuracy of 85% on the test dataset.
- **Custom CNN**: Achieved an accuracy of 88% with optimized hyperparameters.
- **ResNet50 (Transfer Learning)**: Achieved an accuracy of 90% after fine-tuning and pruning.

## Visualizations & Interpretations
Model performance was further analyzed through visualizations:
- **Confusion Matrices**: Displayed the classification results for each model.
- **SHAP Values**: Highlighted key features contributing to the Alzheimer's Disease classification, especially for ResNet50.

## Key Insights
- **Feature Importance**: Certain brain regions, such as the hippocampus, were identified as critical for predicting Alzheimer's disease progression.
- **Model Interpretability**: SHAP values allowed for the identification of misclassifications and validation of the model’s focus on relevant features. <br /> <br />
-->

# File Tree
```
Alzheimers-MRI-Classification/
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 00_data_cleaning
│   ├── 01_random_forest
│   ├── 02_custom_CNN
│   ├── 03_ResNet50_transfer_learning
│   └──04_model_explainability
│
├── src/
│   ├── custom_pruning.py
│   ├── data_io.py
│   ├── img_preprocessing.py
│   ├── random_forest_permute.py
│   └── visualize.py
│
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
```

# Requirements
To reproduce this project, see the dependencies in the requirements.txt file <br /> <br />

# License
This project is licensed under the MIT License - see the LICENSE file for details.
