# Breast Cancer Prediction ML Project

This project uses classical machine learning algorithms to predict breast cancer based on medical data. The goal is to select the best model for early detection and improve diagnostic accuracy.

## Features

- Contains Jupyter Notebook which was used for early experimental analysis 
- Containes modules for data preprocessing, classification_metrics, crossvalidation, gridseach
- Contains scripts for comapring models (Logistic regression and Random Forest) and script to run training and print eval metrics with the best model
- Contains supporting files to install dependencies.


## Dataset

The project uses the [Breast Cancer Wisconsin (Diagnostic) Dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-dataset) available in scikit-learn. I have provided the dataset in the `data/` directory for easy access.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/dKaustav43/breast-cancer_prediction_ml.git
   cd breastcancerpredict_notebook
   ```
2. Install dependencies:
   ```bash
   pip install .
   ```
   or with UV
    ```bash
    uv sync
    ```
## Usage

Open the terminal in the project directory and run the following command to see the see the final model evaluation results:
```bash
python scripts/final_training_and_eval.py
```
To see the model comparison results, run:
```bash
python scripts/model_comparison.py
```
## Using Docker

1. Build the Docker image:
   ```bash
   docker build -t breast-cancer-prediction:1.0 .
   ```
2. Run the Docker container:
    ```bash
    docker run -it --rm breast-cancer-prediction:1.0
    ```

## Results

- Model accuracy and confusion matrix
- Comparison of Logistic Regression and Random Forest models based on evaluation metrics.

## License

This project is licensed under the MIT License.
