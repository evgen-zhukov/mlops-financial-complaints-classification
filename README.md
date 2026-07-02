# mlops-financial-complaints-classification

## Project Overview

This project implements an end-to-end MLOps pipeline for financial complaint classification.

The current version includes:
- dataset preparation and versioning with DVC
- annotation with Label Studio
- experiment tracking and model registry with MLflow
- model training on AWS EC2
- model inference with FastAPI and Docker
- model loading directly from MLflow Model Registry

## Dataset

The source dataset is the public CFPB Consumer Complaint Database.
https://www.consumerfinance.gov/data-research/consumer-complaints/
The preparation script downloads the original CFPB dataset, extracts it, filters selected product categories, and creates a balanced sample.

Selected categories:

* Credit card
* Checking or savings account
* Mortgage
* Debt collection
* Credit reporting or other personal consumer reports

Prepared files:


data/raw/financial_complaints_sample.csv
data/processed/financial_complaints_sample.parquet


The sample contains 200 records, 40 records per category.

The final training dataset used for model training contains 2,500 labeled complaints, balanced across five product categories:

- Credit card — 500
- Checking or savings account — 500
- Mortgage — 500
- Debt collection — 500
- Credit reporting — 500

## Technologies

- Python
- DVC
- MinIO
- MLflow
- Ray
- AWS EC2
- Scikit-learn
- FastAPI
- Docker
- MLflow Model Registry

## Data Preparation

To prepare the dataset:


python scripts/prepare_dataset.py


The script performs the following steps:

1. Downloads the original CFPB complaints dataset.
2. Extracts the CSV file.
3. Filters complaints with non-empty complaint text.
4. Selects five financial product categories.
5. Creates a balanced sample (200 complains).
6. Saves the result as CSV and Parquet.

## Annotation Tool

The annotation tool used in this project is Label Studio.

Label Studio is running on an AWS EC2 instance using Docker.

The annotation task is a single-label text classification task.

Labeling interface:


<View>
  <Text name="complaint" value="$complaint_text"/>

  <Choices name="product_category" toName="complaint" choice="single">
    <Choice value="Credit card"/>
    <Choice value="Checking or savings account"/>
    <Choice value="Mortgage"/>
    <Choice value="Debt collection"/>
    <Choice value="Credit reporting or other personal consumer reports"/>
  </Choices>
</View>


The input file for Label Studio is:


data/raw/financial_complaints_sample.csv


## Annotation Versions

Two annotation exports were created from Label Studio:

data/labeled/project-1-at-2026-06-28-10-14-ddee4add.json
data/labeled/project-1-at-2026-06-28-10-28-804cb3f7.json

Version summary:

* v1: 22 labeled complaints
* v2: 27 labeled complaints

## Dataset Versioning

Dataset versioning is implemented with DVC.

DVC tracks the prepared dataset and annotation exports using `.dvc` metadata files.

Large data files are stored in MinIO, which is used as an S3-compatible object storage.

DVC remote:

s3://mlops-dataset/dvc


## Data Lineage

The data flow is:

CFPB source dataset
        ->
prepare_dataset.py
        ->
balanced CSV sample
        ->
Label Studio annotation
        ->
annotation export JSON
        ->
DVC versioning
        ->
MinIO object storage

## Model Training

Training is executed on AWS EC2.

Example of one of the trainings:
python scripts/train_model.py \
    --run-name baseline \
    --max-features 3000 \
    --ngram-max 1 \
    --c 1.0

Components:

- DVC
- MinIO
- Ray
- MLflow Tracking
- MLflow Model Registry

## Model API Inference

The inference service is implemented using FastAPI and Docker.

The application loads the model directly from the MLflow Model Registry during startup. The model is not stored inside the GitHub repository.

# Build Docker Image

docker build -t financial-complaints-api .


# Run Inference Service

docker run -d \
  --name complaints-api \
  --restart unless-stopped \
  --network host \
  -e MLFLOW_TRACKING_URI=http://localhost:5000 \
  -v /home/ubuntu/mlops-hw2/mlops-financial-complaints-classification/artifacts:/home/ubuntu/mlops-hw2/mlops-financial-complaints-classification/artifacts \
  financial-complaints-api

  Note: MLflow in this project uses a local artifact store on the EC2 instance. Because MLflow stores model artifact paths as local filesystem paths, the `artifacts` directory is mounted into the Docker container. The model itself is not committed to GitHub.

# API Documentation

After the service starts, Swagger UI is available at:
http://51.20.96.49:8000/docs


# Health Check
GET /health

Example response:

{
  "status": "healthy",
  "model_loaded": true,
  "model_uri": "models:/financial_complaints_classifier/2"
}

# Prediction
POST /predict

Example request:

{
  "complaint_text": "My credit card was charged twice and I want this transaction refunded."
}

Example response:

{
  "predicted_category": "Credit card"
}

## MLflow Model Registry

The inference service loads the model from MLflow Model Registry using the following model URI:

models:/financial_complaints_classifier/2

