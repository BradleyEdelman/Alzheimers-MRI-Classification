## Alzheimer's Disease Classification from Anatomical MRI Images <br /> <br />

### Project Overview

This project aims to classify four stages of Alzheimer's Disease (AD) from anatomical MRI images. The project explores various machine learning techniques, including traditional models like Random Forest, and deep learning approaches such as CNNs and transfer learning.

#### Highlights:
- **Data Exploration & Preprocessing**: Checking data quality, dimensionality reduction via PCA, and addressing class imbalances.
- **Modeling & Evaluation**: Training and evaluating multiple ML models, with complementary techniques such as permutation testing, hyperparameter tuning, and transfer learning.
- **Model Explainability**: Using PCA maps and SHAP values to interpret model predictions and identify key features of AD progression.
- **Infrastructure**: Executed on Databricks with AWS EC2 for scalable computing and parallel processing.

### Dataset
The dataset used for this project contains labeled anatomical MRI images for AD classification, and is available on Kaggle:
[Alzheimer MRI Disease Classification Dataset](https://www.kaggle.com/datasets/borhanitrash/alzheimer-mri-disease-classification-dataset) <br /> <br />

## Notebooks
### 00 Data Cleaning
Prepares the dataset for classification by:
- Accessing Parquet data from an AWS S3 bucket
- Addressing class imbalance using synthetic data generation techniques
- Formatting and standardizing data for evaluation and cross-validation <br />

### 01 Random Forest
Utilizes a Random Forest to classify AD, with steps that include:
- Applying Principal Component Analysis (PCA) for dimensionality reduction and feature extraction
- Cross-validation of classification results
- Permutation testing for statistical significance using parallel processing
- Exploring feature importance <br />

### 02 Custom CNN
Develops a custom convolutional neural network (CNN) to classify AD, focusing on:
- Hyperparameter tuning to optimize model performance
- Distributed training using TensorFlow's MirroredStrategy
- The effect of class imbalance on class-specific classification accuracy <br />

### 03 ResNet50 Transfer Learning
Implements transfer learning with the ResNet50 model, exploring:
- Fine-tuning to improve classification accuracy
- Prediction accuracy and complexity metrics as a function of model pruning <br />

### 04 Model Explainability
Explores the explainability of the fine-tuned ResNet50 model using SHapley Additive exPlanations (SHAP):
- Identifies key spatial features of Alzheimer's progression
- Investigates reasons for misclassifications
- Performs cluster analyses <br /> <br />

## Results <br />

### Model Performance
This section briefly summarizes the performance of the different models and approaches used in this project:

- **Random Forest**: Achieved a maximum accuracy of 91.59% when using 50 PC features, but was consistently above 90% with as few as 30 PC features.
- **Custom CNN**: Achieved an accuracy of 95.55% with optimized hyperparameters, but suffered from class-specific biases.
- **ResNet50 (Transfer Learning)**: Achieved an accuracy of 95.86% after fine-tuning the ResNet50 model, and produced more explainable misclassification instances (i.e. confusion matrices).

### Key Insights
- **Model Interpretability**: Spatial PC maps and SHAP values facilitated the identification of brain areas implicated in the progression of AD. These visualization tools also validated the models' focus on relevant features available via non-invasive MRI. <br /> <br />

## File Tree
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
│   └── 04_model_explainability
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

## Requirements
To reproduce this project, see the dependencies in the requirements.txt file <br /> <br />

## License
This project is licensed under the MIT License - see the LICENSE file for details.
