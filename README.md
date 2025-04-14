## Alzheimer's Disease Classification from Anatomical MRI Images <br /> <br />

### Project Overview

This project aims to classify four stages of Alzheimer's Disease (AD) from anatomical MRI images. This data was obtained from an open source dataset containing images of individual slices rather than whole-brain information. Furthermore, the data was provided in what I assume is a pre-processed and registered format, however, the toolboxes and steps used to process the raw dicom/nifti files is unknown to me. Nevertheless, I think this is still a decent open-source dataset to demonstrate various machine learning techniques, including traditional models like Random Forest, and deep learning approaches such as CNNs and transfer learning.

#### Highlights:
- **Data Exploration & Preprocessing**: Checking data quality, dimensionality reduction via PCA, and examining potential solutions to class imbalance.
- **Modeling & Evaluation**: Training and evaluating multiple ML models, with complementary techniques such as class weighting, permutation testing, hyperparameter tuning, and transfer learning.
- **Model Explainability**: Using PCA maps and SHAP values to interpret model predictions and identify key features of AD progression.
- **Infrastructure**: Executed on Databricks with AWS EC2 for scalable computing and parallel processing.

### Dataset
The dataset used for this project contains labeled anatomical MRI images for AD classification, and is available on Kaggle:
[Alzheimer MRI Disease Classification Dataset](https://www.kaggle.com/datasets/borhanitrash/alzheimer-mri-disease-classification-dataset) <br /> <br />

## Notebooks
### 00 Data Cleaning
Prepares the dataset for classification by:
- Accessing Parquet data from an AWS S3 bucket
- Formatting and standardizing training data
- Assessing different approaches to address class imbalance (e.g. synthetic data generation, augmentation)
- Preprocessing test data for evaluation and cross-validation <br />

### 01 Random Forest
Utilizes a Random Forest to classify AD, with steps that include:
- Applying Principal Component Analysis (PCA) for dimensionality reduction and feature extraction
- Cross-validation of classification results
- Hyperparameter tuning
- Permutation testing for statistical significance using parallel processing
- Exploring feature importance <br />

### 02 Custom CNN
Develops a custom convolutional neural network (CNN) to classify AD, focusing on:
- Class weighting to address class imbalance
- Hyperparameter tuning
- Distributed training using TensorFlow's MirroredStrategy
- The effect of class imbalance on class-specific classification accuracy <br />

### 03 ResNet50 Transfer Learning
Implements transfer learning with the ResNet50 model, exploring:
- Fine-tuning the model to improve classification accuracy (with class weighting)
- Prediction accuracy and complexity metrics as a function of model pruning <br />

### 04 Model Explainability
Explores the explainability of the fine-tuned ResNet50 model using SHapley Additive exPlanations (SHAP):
- Identifies key spatial features of Alzheimer's progression
- Investigates reasons for misclassifications
- Performs cluster analyses <br /> <br />

## Results <br />

### Model Performance
This section briefly summarizes the performance of the different models and approaches used in this project:

- **Random Forest**: Exhibited a plateauing increase in classification accuracy as a function of PC features used. Accuracy was consistently above 90% with when using ~30-50 features.
- **Custom CNN**: Achieved an accuracy of >96% with optimized hyperparameters and class weighting. While these two techniques did not dramatically improve or change over accuracy, they did help alleviate class-specific biases.
- **ResNet50 (Transfer Learning)**: Achieved an accuracy of >96% after fine-tuning the ResNet50 model with class weighting, however, resulted in lower class-specific F1 scores compared to the class-weighted and fine tuned custom CNN.

### Key Insights
- **Model Interpretability**: Spatial PC maps and SHAP values facilitated the identification of brain areas implicated in the progression of AD. These visualization tools also validated the models' focus on relevant features available via non-invasive MRI. <br /> <br />

## File Tree
```
Alzheimers-MRI-Classification/
├── data/
│   └── raw/
│
├── notebooks_databricks/
│   ├── 00_data_cleaning.ipynb
│   ├── 01_random_forest.ipynb
│   ├── 02_custom_CNN.ipynb
│   ├── 03_ResNet50_transfer_learning.ipynb
│   └── 04_model_explainability.ipynb
│
├── notebooks_jupyter/
│   ├── 00_data_cleaning_jupyter.ipynb
│   ├── 01_random_forest_jupyter.ipynb
│   ├── 02_custom_CNN_jupyter.ipynb
│   ├── 03_ResNet50_transfer_learning_jupyter.ipynb
│   └── 04_model_explainability_jupyter.ipynb
│
├── src/
│   ├── custom_pruning.py
│   ├── data_io.py
│   ├── img_preprocessing.py
│   ├── random_forest_permute.py
│   └── visualize.py
│
├── .dockerignore
├── .gitattributes
├── .gitignore
├── dockerfile
├── LICENSE
├── README.md
├── requirements_databricks.txt
├── requirements_jupyter.txt
```

## Running with Docker
To run this project in an isolated environment using Docker:

### Build the image:
```bash
docker build -t ad-mri-classification .
```

### Run the container:
```bash
docker run -p 8888:8888 ad-mri-classification
```

This launches a Jupyter notebook server at http://localhost:8888 where you can explore and run the notebooks_jupyter directory

Notes:
- The container installs all necessary dependencies using `requirements_jupyter.txt`
- The data/ folder is included in the image since the raw parquet files are quite small.

## Requirements
To reproduce this project, see the dependencies in the requirements_jupyter.txt (and/or requirements_databricks.txt) file <br /> <br />

## License
This project is licensed under the MIT License - see the LICENSE file for details.
