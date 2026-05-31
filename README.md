# Credit Card Fraud Detection

This project explores credit card fraud detection using a real-world transaction dataset and several supervised machine learning models. The analysis is implemented in [analysis.ipynb](analysis.ipynb) and the trained model artifact is saved as [fraud_model.joblib](https://huggingface.co/Santosh-Chapagain/fraud-card-model).

In addition to local model development, this project was operationalized on Azure Machine Learning by building an end-to-end ML pipeline and publishing a REST endpoint for real-time HTTP inference requests.

## Overview

Credit card fraud detection is a highly imbalanced classification problem, where fraudulent transactions are rare compared with legitimate ones. This notebook-based project covers:

- data loading and inspection
- preprocessing and feature scaling
- handling class imbalance with resampling techniques
- training and comparing multiple models
- saving a trained model for later reuse
- deploying the trained workflow through an Azure ML pipeline and exposing it as a REST API endpoint

## Dataset

The project uses [creditcard.csv](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), which contains transaction features and a binary target column indicating whether a transaction is fraudulent.

## Models Used

The notebook includes experiments with:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

It also explores imbalance-handling strategies such as undersampling and SMOTE-style resampling.

## Project Structure

- [analysis.ipynb](analysis.ipynb) - main notebook for analysis, training, and evaluation
- [creditcard.csv](creditcard.csv) - input dataset
- [fraud_model.joblib](fraud_model.joblib) - saved trained model

## Requirements

Typical Python packages used in this project include:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- imbalanced-learn
- joblib

## How to Run

1. Open [analysis.ipynb](analysis.ipynb) in VS Code or Jupyter.
2. Run the cells from top to bottom.
3. Review the evaluation results for the different models and sampling approaches.

If you want to use the saved model directly:

```python
import joblib

model = joblib.load("fraud_model.joblib")
```

## Make Predictions

After loading the model, pass a feature matrix with the same preprocessing steps used during training:

```python
predictions = model.predict(X_new)
```

If the training notebook applied scaling or resampling, apply the same preprocessing pipeline before prediction.

## Azure ML Pipeline and REST Deployment

This project also includes a production-style deployment workflow in Azure:

- Built and validated an Azure Machine Learning pipeline for data preparation, model training, and model registration.
- Deployed the trained model as a managed online endpoint.
- Successfully published a REST API that accepts HTTP requests and returns fraud prediction responses.

Example request flow:

1. Prepare a JSON payload with transaction feature values.
2. Send an authenticated HTTP POST request to the Azure endpoint URI.
3. Receive prediction output (fraud or non-fraud) from the endpoint response.

Example (conceptual) HTTP request:

```bash
curl -X POST "<azure-endpoint-uri>" \
	-H "Content-Type: application/json" \
	-H "Authorization: Bearer <access-token-or-key>" \
	-d '{"input_data": [{"Time": 0.0, "V1": -1.36, "V2": -0.07, "Amount": 149.62}]}'
```

## Notes

- This is an imbalanced classification problem, so accuracy alone is not the best metric.
- Precision, recall, F1-score, and confusion matrix are more useful for evaluating fraud detection models.
- Keep the preprocessing steps consistent between training and inference.

## License

No license has been specified yet.